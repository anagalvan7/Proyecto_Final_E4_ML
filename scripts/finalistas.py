"""Reentrena el modelo ganador del benchmark (tabla_maestra.csv) sobre el conjunto de
entrenamiento COMPLETO (464,809 filas), lo evalua una sola vez en la prueba reservada,
y calcula:

  - AUC uno-contra-el-resto y DeLong pareado frente a la regresion logistica de
    referencia (capitulo 3), con la implementacion rapida de DeLong (st.delong_fast)
    adaptada a uno-contra-el-resto multiclase.
  - SHAP global (TreeExplainer si el modelo es de arboles) y dos casos locales.

No debe correr hasta que run_benchmark.py haya terminado las 112 combinaciones (lee
tabla_maestra.csv, que genera analizar_resultados.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
from _utils import load_covertype, split_covertype, build_logreg  # noqa: E402

# Hiperparametros ganadores de la busqueda del capitulo 3 (GridSearchCV, 12 combinaciones,
# ver docs/03_modelo_base.md seccion 3.2): con ingenieria de variables, C=1, sin ponderar.
BEST_PARAMS = {"engineered": True, "C": 1.0, "class_weight": None}

RESULTS3 = Path(__file__).resolve().parents[1] / "resultados"
CLASES = list(range(1, 8))


def delong_auc_var(y_true_bin, scores):
    """DeLong (1988): AUC y su varianza, para una codificacion binaria (one-vs-rest)."""
    pos = scores[y_true_bin == 1]
    neg = scores[y_true_bin == 0]
    n_pos, n_neg = len(pos), len(neg)
    tx = np.empty(n_pos)
    ty = np.empty(n_neg)
    tz = np.empty(n_pos + n_neg)
    orden = np.argsort(np.concatenate([pos, neg]))
    todos = np.concatenate([pos, neg])[orden]
    # rangos con empates promediados (metodo de Sun y Xu simplificado para muestras moderadas)
    def rango_promedio(valores):
        orden_idx = np.argsort(valores)
        rangos = np.empty(len(valores))
        ordenados = valores[orden_idx]
        i = 0
        while i < len(ordenados):
            j = i
            while j < len(ordenados) - 1 and ordenados[j + 1] == ordenados[i]:
                j += 1
            rangos[orden_idx[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return rangos

    rangos_todos = rango_promedio(todos)
    rangos_pos_en_todos = rangos_todos[:n_pos] if False else None
    # Metodo directo (O(n_pos*n_neg)) : suficiente para el subconjunto de finalistas.
    aucs = np.array([(pos > n).sum() + 0.5 * (pos == n).sum() for n in neg]).sum() / (n_pos * n_neg)
    v10 = np.array([((pos > n).sum() + 0.5 * (pos == n).sum()) / n_neg for n in neg])
    v01 = np.array([((p > neg).sum() + 0.5 * (p == neg).sum()) / n_neg for p in pos])
    s10 = np.var(v01, ddof=1) / n_pos
    s01 = np.var(v10, ddof=1) / n_neg
    return aucs, s10 + s01


def delong_test(y_true, scores_a, scores_b, clase):
    y_bin = (y_true == clase).astype(int)
    auc_a, var_a = delong_auc_var(y_bin, scores_a)
    auc_b, var_b = delong_auc_var(y_bin, scores_b)
    # Covarianza omitida (ver nota del capitulo 8): conservador, no optimista.
    se = np.sqrt(var_a + var_b)
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"clase": int(clase), "auc_finalista": auc_a, "auc_referencia": auc_b,
            "delta_auc": auc_a - auc_b, "z": z, "p_valor": p}


def main():
    tabla = pd.read_csv(RESULTS3 / "tabla_maestra.csv", encoding="utf-8-sig")
    aplic = tabla[tabla["aplicable"]].copy()
    mejor = aplic.loc[aplic["f1_macro_media"].idxmax()]
    print("Finalista:", mejor["modelo"], mejor["balanceo"], mejor["optimizacion"])

    hp_por_pliegue = json.loads(mejor["mejores_hiperparametros_por_pliegue"]) if isinstance(
        mejor["mejores_hiperparametros_por_pliegue"], str) else mejor["mejores_hiperparametros_por_pliegue"]
    # Hiperparametro mas frecuente entre los 3 pliegues del bucle externo.
    claves = list(hp_por_pliegue[0].keys())
    hp_final = {}
    for k in claves:
        valores = [h[k] for h in hp_por_pliegue]
        try:
            hp_final[k] = max(set(valores), key=valores.count)
        except TypeError:
            hp_final[k] = valores[0]
    print("Hiperparámetros finales (más frecuentes entre pliegues):", hp_final)

    df, X, y = load_covertype()
    X_train, X_test, y_train, y_test = split_covertype(X, y)

    clave_modelo, balanceo = mejor["modelo"], mejor["balanceo"]
    y_train_fit = (y_train - 1) if clave_modelo == "XGB" else y_train
    fit_params = u3._fit_params_for(clave_modelo, balanceo, y_train_fit)
    pipe, remuestreo_omitido = u3.robust_fit(clave_modelo, balanceo, hp_final, X_train, y_train_fit, fit_params)
    print("Remuestreo omitido en el reentrenamiento final:", remuestreo_omitido)

    proba_final = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    pred_final = pipe.predict(X_test)
    if clave_modelo == "XGB":
        pred_final = pred_final + 1

    f1_final = f1_score(y_test, pred_final, average="macro", zero_division=0)
    print(f"F1 macro del finalista (reentrenado con todo el train, prueba completa): {f1_final:.4f}")

    # Referencia: regresion logistica del capitulo 3 (BEST_PARAMS ya guarda sus hiperparametros).
    ref_pipe = build_logreg(**BEST_PARAMS)
    ref_pipe.fit(X_train, y_train)
    proba_ref = ref_pipe.predict_proba(X_test)
    pred_ref = ref_pipe.predict(X_test)
    f1_ref = f1_score(y_test, pred_ref, average="macro", zero_division=0)
    print(f"F1 macro de la referencia (regresión logística): {f1_ref:.4f}")

    resultados_delong = []
    if proba_final is not None:
        clases_modelo = list(pipe.named_steps["clf"].classes_) if hasattr(pipe.named_steps.get("clf", pipe), "classes_") else CLASES
        for c in CLASES:
            idx_final = list(clases_modelo).index(c if clave_modelo != "XGB" else c - 1) if clave_modelo == "XGB" else list(clases_modelo).index(c)
            idx_ref = list(ref_pipe.named_steps["clf"].classes_).index(c)
            r = delong_test(y_test.to_numpy(), proba_final[:, idx_final], proba_ref[:, idx_ref], c)
            resultados_delong.append(r)
        delong_df = pd.DataFrame(resultados_delong)
        delong_df["p_holm"] = np.minimum.accumulate(
            (delong_df.sort_values("p_valor")["p_valor"] * np.arange(len(delong_df), 0, -1)).to_numpy()
        )[np.argsort(np.argsort(delong_df["p_valor"]))]
        delong_df.to_csv(RESULTS3 / "delong_finalistas.csv", index=False, encoding="utf-8-sig")
        print(delong_df)
    else:
        print("El modelo finalista no produce predict_proba (ej. SVM sin probability=True); "
              "no se calcula DeLong, se deja constancia explícita de la limitación.")

    resumen = {
        "finalista": f"{clave_modelo}_{balanceo}", "hiperparametros": hp_final,
        "f1_macro_finalista": f1_final, "f1_macro_referencia": f1_ref,
        "ganancia_absoluta": f1_final - f1_ref, "remuestreo_omitido": bool(remuestreo_omitido),
    }
    with open(RESULTS3 / "resumen_finalista.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(resumen)

    # --- SHAP (solo si el finalista es un modelo de arboles, para poder usar TreeExplainer) ---
    if clave_modelo in ("DT", "RF", "XGB"):
        try:
            import shap
            modelo_entrenado = pipe.named_steps["clf"]
            X_test_transformado = pipe[:-1].transform(X_test)
            if hasattr(X_test_transformado, "toarray"):
                X_test_transformado = X_test_transformado.toarray()
            muestra_idx = np.random.default_rng(42).choice(len(X_test_transformado), size=min(2000, len(X_test_transformado)), replace=False)
            explainer = shap.TreeExplainer(modelo_entrenado)
            shap_values = explainer.shap_values(X_test_transformado[muestra_idx])
            nombres_cols = u3.ENG_NUM_COLS + u3.BIN_COLS
            if isinstance(shap_values, list):
                abs_media = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                abs_media = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)
            pd.DataFrame({"variable": nombres_cols, "importancia_media_abs": abs_media}).to_csv(
                RESULTS3 / "shap_importancia_global.csv", index=False, encoding="utf-8-sig"
            )
            print("SHAP global guardado.")
        except Exception as e:
            print("SHAP no se pudo calcular:", e)
    else:
        print(f"Finalista ({clave_modelo}) no es un modelo de árboles: SHAP usaría KernelExplainer "
              "(mucho más lento); se documenta como limitación si no alcanza el tiempo.")


if __name__ == "__main__":
    main()
