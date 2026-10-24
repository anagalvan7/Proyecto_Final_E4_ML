"""Reentrenamiento final, serializacion y evaluacion en la prueba reservada (bloque B).

1. Cada una de las 7 mejores combinaciones (seccion 4.1 de la guia) se reentrena con la
   configuracion modal de sus pliegues externos sobre la muestra de entrenamiento del
   benchmark (40.000 filas estratificadas; ver capitulo 3 del libro para la justificacion).
2. El ECLC se ajusta sobre las probabilidades fuera de pliegue de esos 7 modelos en las
   mismas 40.000 filas; (k, beta, tau) se eligen con 6 ensayos de TPE, como en el CV.
3. HistGradientBoosting (competidor del estado del arte) con Random Search de 6 configs.
4. Todos se evaluan UNA vez en la prueba reservada (116.203 filas, seccion 1.6).
5. Se serializa cada modelo con joblib (compress=3) en dashboard/artifacts/, con un
   metadata.json, y se verifica la prueba de ida y vuelta (predicciones identicas).

Salidas: dashboard/artifacts/*, resultados/e3/holdout_*.
"""
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import imblearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline

import _e3 as e3
import _utils3 as u3
from src import preprocesamiento
from src.original_model import EnsambleCompetenciaLocal

# Los pipelines serializados deben referenciar la ingenieria de variables del paquete del
# dashboard (identica a la del libro) para cargarse en Render sin el resto del proyecto.
u3.engineer_features = preprocesamiento.engineer_features

ART = e3.RAIZ / "dashboard" / "artifacts"
ART.mkdir(exist_ok=True)
OUT = e3.RES_E3


def esquema_entrada(X):
    """Nombres, tipos y rangos validos de las 54 variables de entrada (para el formulario)."""
    esquema = []
    for c in preprocesamiento.FEATURES:
        binaria = c.startswith(("Wilderness", "Soil"))
        esquema.append({"nombre": c, "tipo": "binaria" if binaria else "entera",
                        "min": int(X[c].min()), "max": int(X[c].max())})
    return esquema


def metricas(y, P):
    pred = e3.CLASES[P.argmax(1)]
    return {"f1_macro": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
            "exactitud_balanceada": balanced_accuracy_score(y, pred),
            "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=e3.CLASES),
            "f1_por_clase": f1_score(y, pred, average=None, labels=e3.CLASES).tolist()}


def main():
    X, y, X_test, y_test, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    y_np, y_test_np = y.to_numpy(), y_test.to_numpy()
    mejores = e3.mejores_combinaciones()
    tabla = e3.tabla_maestra()
    versiones = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                 "xgboost": xgboost.__version__, "imbalanced-learn": imblearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__}
    esquema = esquema_entrada(pd.concat([X, X_test]))
    X_ida_vuelta = X_test.iloc[:2000]

    P_test, P_oof, resumen, pipes = {}, [], [], {}
    for _, fila in mejores.iterrows():
        m, b, o = fila["modelo"], fila["balanceo"], fila["optimizacion"]
        params = fila["params_moda"]
        t0 = time.perf_counter()
        P_oof.append(e3.probas_oof(m, b, params, X, y))
        t1 = time.perf_counter()
        pipe = e3.ajustar_modelo(m, b, params, X, y)
        t2 = time.perf_counter()
        P_test[m] = e3.proba_ordenada(pipe, m, X_test)
        t3 = time.perf_counter()
        ruta = ART / f"{m}.joblib"
        joblib.dump(pipe, ruta, compress=3)
        recargado = joblib.load(ruta)
        ida_vuelta = bool(np.array_equal(pipe.predict(X_ida_vuelta), recargado.predict(X_ida_vuelta))
                          and np.allclose(pipe.predict_proba(X_ida_vuelta), recargado.predict_proba(X_ida_vuelta)))
        met = metricas(y_test_np, P_test[m])
        fila_tabla = tabla[(tabla.modelo == m) & (tabla.balanceo == b) & (tabla.optimizacion == o)].iloc[0]
        resumen.append({
            "modelo": m, "nombre": u3.NOMBRES_MODELOS[m], "balanceo": b, "optimizacion": o,
            "hiperparametros": params, "f1_cv_media": float(fila["f1_macro_media"]),
            "f1_cv_sd": float(fila["f1_macro_sd"]), "f1_cv_pliegues": json.loads(fila_tabla["f1_macro_pliegues"]),
            "tiempo_ajuste_s": t2 - t1, "tiempo_inferencia_s_por_1000": (t3 - t2) / len(X_test) * 1000,
            "tamano_mb": ruta.stat().st_size / 2**20, "ida_y_vuelta_ok": ida_vuelta,
            "prueba_reservada": met})
        json.dump({"modelo": m, "nombre": u3.NOMBRES_MODELOS[m], "balanceo": b, "optimizacion": o,
                   "hiperparametros_finales": params, "semilla": u3.RANDOM_STATE, "versiones": versiones,
                   "datos_entrenamiento": f"submuestra estratificada de {len(X)} filas del entrenamiento"
                                          + (f" (SVM: {e3.N_SVM_MAX} filas)" if m == "SVM" else ""),
                   "esquema_entrada": esquema, "clases": e3.CLASES.tolist(),
                   "metricas_bucle_externo": {"f1_macro_media": float(fila["f1_macro_media"]),
                                              "f1_macro_sd": float(fila["f1_macro_sd"]),
                                              "f1_macro_pliegues": json.loads(fila_tabla["f1_macro_pliegues"])},
                   "metricas_prueba_reservada": met, "ida_y_vuelta_ok": ida_vuelta},
                  open(ART / f"{m}.metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        pipes[m] = pipe
        print(f"{m}: F1 prueba={met['f1_macro']:.4f} tamano={resumen[-1]['tamano_mb']:.1f} MB "
              f"ida/vuelta={ida_vuelta}", flush=True)

    # ------------------------------------------------------------ ECLC
    P_oof = np.stack(P_oof)
    from e3_meta import optimizar
    import e3_meta
    t0 = time.perf_counter()
    hp, historia = optimizar(X, y_np, P_oof)
    eclc = EnsambleCompetenciaLocal(
        estimadores=[(m, None) for m in mejores["modelo"]], random_state=u3.RANDOM_STATE,
        transformador_vecinos=Pipeline(u3.build_preprocessor_steps()), **hp,
    ).ajustar_desde_oof(X, y_np, P_oof)
    t1 = time.perf_counter()
    P_base_test = np.stack([P_test[m] for m in mejores["modelo"]])
    P_test["ECLC"] = eclc.combinar(P_base_test, X=X_test)
    t2 = time.perf_counter()
    ruta = ART / "ECLC.joblib"
    joblib.dump(eclc, ruta, compress=3)
    rec = joblib.load(ruta)
    ida_vuelta = bool(np.allclose(rec.combinar(P_base_test[:, :2000], X=X_ida_vuelta), P_test["ECLC"][:2000]))
    met = metricas(y_test_np, P_test["ECLC"])
    resumen.append({"modelo": "ECLC", "nombre": "Ensamble por competencia local y por clase (original)",
                    "balanceo": "pesos de clase en la vecindad", "optimizacion": "bayes",
                    "hiperparametros": hp, "tiempo_ajuste_s": t1 - t0,
                    "tiempo_inferencia_s_por_1000": (t2 - t1) / len(X_test) * 1000,
                    "tamano_mb": ruta.stat().st_size / 2**20, "ida_y_vuelta_ok": ida_vuelta,
                    "prueba_reservada": met, "historia_optuna": historia})
    json.dump({"modelo": "ECLC", "hiperparametros_finales": hp, "semilla": u3.RANDOM_STATE,
               "versiones": versiones, "modelos_base": mejores["modelo"].tolist(),
               "datos_entrenamiento": f"probabilidades fuera de pliegue (3 pliegues) de los 7 modelos base "
                                      f"sobre {len(X)} filas", "esquema_entrada": esquema,
               "clases": e3.CLASES.tolist(), "metricas_prueba_reservada": met, "ida_y_vuelta_ok": ida_vuelta,
               "uso": "P = combinar(np.stack([proba de cada modelo base en el orden modelos_base]), X=X)"},
              open(ART / "ECLC.metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ECLC: hp={hp} F1 prueba={met['f1_macro']:.4f} ida/vuelta={ida_vuelta}", flush=True)

    # ------------------------------------------------------------ HGB (no se despliega)
    from e3_probas_pliegue import hgb_random_search
    p_hgb = hgb_random_search(X, y)
    hgb = HistGradientBoostingClassifier(random_state=u3.RANDOM_STATE, early_stopping=False, **p_hgb).fit(X, y)
    P_test["HGB"] = hgb.predict_proba(X_test)
    resumen.append({"modelo": "HGB", "nombre": "HistGradientBoosting (estado del arte)",
                    "hiperparametros": p_hgb, "prueba_reservada": metricas(y_test_np, P_test["HGB"])})

    # ------------------------------------------------------------ salidas
    pd.DataFrame(resumen).to_json(OUT / "holdout_resumen.json", orient="records", force_ascii=False, indent=1)
    np.savez_compressed(OUT / "holdout_probas.npz", y_test=y_test_np,
                        **{k: v.astype(np.float32) for k, v in P_test.items()})
    for k, P in P_test.items():
        cm = confusion_matrix(y_test_np, e3.CLASES[P.argmax(1)], labels=e3.CLASES)
        pd.DataFrame(cm, index=e3.CLASES, columns=e3.CLASES).to_csv(OUT / f"holdout_confusion_{k}.csv")
    print(pd.DataFrame([{"modelo": r["modelo"], **{k: v for k, v in r["prueba_reservada"].items()
                                                   if k != "f1_por_clase"}} for r in resumen]).round(4).to_string())


if __name__ == "__main__":
    main()
