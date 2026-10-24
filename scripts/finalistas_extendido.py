"""Extiende finalistas.py con la evaluacion completa del finalista:

- Matriz de confusion, curvas ROC/PR y tabla completa de metricas del finalista.
- Calibracion (Brier multiclase, ECE por clase, diagrama de confiabilidad).
- Intervalos de confianza por bootstrap (finalista y referencia).
- Tamano del efecto: delta de Cliff sobre las distribuciones bootstrap de F1 macro.
- SHAP local (2 observaciones) y comparacion con LIME sobre las mismas 2.
- Perfilamiento: XGBoost estandar (sin ajustar) vs optimizado (tiempo, memoria, desempeno).
- Sensibilidad a la semilla: 5 semillas, con subsample/colsample=0.8 para que la
  semilla tenga un efecto real (los hiperparametros ganadores no muestrean filas/columnas
  por defecto, asi que sin esto el resultado seria identico entre semillas).

No reentrena los 112 de la rejilla; solo trabaja con el finalista y la referencia.
"""
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
from _utils import load_covertype, split_covertype, build_logreg  # noqa: E402

RESULTS3 = Path(__file__).resolve().parents[1] / "resultados"
CLASES = list(range(1, 8))
BEST_PARAMS_LR = {"engineered": True, "C": 1.0, "class_weight": None}


def cargar_finalista():
    tabla = pd.read_csv(RESULTS3 / "tabla_maestra.csv", encoding="utf-8-sig")
    aplic = tabla[tabla["aplicable"]].copy()
    mejor = aplic.loc[aplic["f1_macro_media"].idxmax()]
    hp_por_pliegue = json.loads(mejor["mejores_hiperparametros_por_pliegue"])
    claves = list(hp_por_pliegue[0].keys())
    hp_final = {}
    for k in claves:
        valores = [h[k] for h in hp_por_pliegue]
        hp_final[k] = max(set(valores), key=valores.count)
    return mejor["modelo"], mejor["balanceo"], hp_final


def main():
    clave_modelo, balanceo, hp_final = cargar_finalista()
    print("Finalista:", clave_modelo, balanceo, hp_final)

    df, X, y = load_covertype()
    X_train, X_test, y_train, y_test = split_covertype(X, y)
    y_train_fit = (y_train - 1) if clave_modelo == "XGB" else y_train
    fit_params = u3._fit_params_for(clave_modelo, balanceo, y_train_fit)

    pipe, _ = u3.robust_fit(clave_modelo, balanceo, hp_final, X_train, y_train_fit, fit_params)
    proba = pipe.predict_proba(X_test)
    pred = pipe.predict(X_test)
    if clave_modelo == "XGB":
        pred = pred + 1
    clases_modelo_orden = [int(c) + 1 if clave_modelo == "XGB" else int(c) for c in pipe.named_steps["clf"].classes_]
    orden_idx = [clases_modelo_orden.index(c) for c in CLASES]
    proba = proba[:, orden_idx]  # reordena columnas a 1..7 consistentemente

    ref_pipe = build_logreg(**BEST_PARAMS_LR)
    ref_pipe.fit(X_train, y_train)
    proba_ref = ref_pipe.predict_proba(X_test)
    pred_ref = ref_pipe.predict(X_test)
    clases_ref_orden = list(ref_pipe.named_steps["clf"].classes_)
    orden_idx_ref = [clases_ref_orden.index(c) for c in CLASES]
    proba_ref = proba_ref[:, orden_idx_ref]

    # ---------- 1. Matriz de confusion, ROC/PR, tabla de metricas ----------
    def tabla_metricas(y_true, pred_, proba_, nombre):
        return {
            "modelo": nombre,
            "Accuracy": accuracy_score(y_true, pred_),
            "Exactitud balanceada": balanced_accuracy_score(y_true, pred_),
            "Precisión macro": precision_score(y_true, pred_, average="macro", zero_division=0),
            "Recall macro": recall_score(y_true, pred_, average="macro", zero_division=0),
            "F1 macro": f1_score(y_true, pred_, average="macro", zero_division=0),
            "F1 ponderado": f1_score(y_true, pred_, average="weighted", zero_division=0),
            "AUC OvR macro": roc_auc_score(y_true, proba_, multi_class="ovr", average="macro"),
            "Log loss": log_loss(y_true, proba_, labels=CLASES),
        }
    metricas_tabla = pd.DataFrame([
        tabla_metricas(y_test, pred, proba, "XGBoost (finalista)"),
        tabla_metricas(y_test, pred_ref, proba_ref, "Regresión logística (referencia)"),
    ])
    metricas_tabla.to_csv(RESULTS3 / "finalista_metricas.csv", index=False, encoding="utf-8-sig")
    print(metricas_tabla)

    cm = confusion_matrix(y_test, pred, labels=CLASES, normalize="true")
    pd.DataFrame(cm, index=CLASES, columns=CLASES).to_csv(RESULTS3 / "finalista_matriz_confusion.csv", encoding="utf-8-sig")

    y_bin = label_binarize(y_test, classes=CLASES)
    roc_data = []
    for i, c in enumerate(CLASES):
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        roc_data.append({"clase": c, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                          "precision": prec.tolist(), "recall": rec.tolist()})
    with open(RESULTS3 / "finalista_roc_pr.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f)
    print("Matriz de confusión y curvas ROC/PR guardadas.")

    # ---------- 2. Calibracion ----------
    calib_rows = []
    ece_por_clase = {}
    for i, c in enumerate(CLASES):
        frac_pos, mean_pred = calibration_curve(y_bin[:, i], proba[:, i], n_bins=10, strategy="quantile")
        for fp, mp in zip(frac_pos, mean_pred):
            calib_rows.append({"clase": c, "prob_predicha": mp, "frecuencia_observada": fp})
        bins = np.quantile(proba[:, i], np.linspace(0, 1, 11))
        bins[-1] += 1e-9
        idx_bin = np.digitize(proba[:, i], bins[1:-1])
        ece = 0.0
        for b in range(10):
            mask = idx_bin == b
            if mask.sum() == 0:
                continue
            conf = proba[mask, i].mean()
            acc = y_bin[mask, i].mean()
            ece += (mask.sum() / len(y_test)) * abs(conf - acc)
        ece_por_clase[c] = ece
    pd.DataFrame(calib_rows).to_csv(RESULTS3 / "finalista_calibracion.csv", index=False, encoding="utf-8-sig")
    brier_multiclase_final = np.mean(np.sum((proba - y_bin) ** 2, axis=1))
    brier_multiclase_ref = np.mean(np.sum((proba_ref - y_bin) ** 2, axis=1))
    resumen_calib = {
        "brier_multiclase_finalista": float(brier_multiclase_final),
        "brier_multiclase_referencia": float(brier_multiclase_ref),
        "ece_por_clase_finalista": {int(k): float(v) for k, v in ece_por_clase.items()},
        "ece_medio_finalista": float(np.mean(list(ece_por_clase.values()))),
    }
    with open(RESULTS3 / "finalista_calibracion_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen_calib, f, ensure_ascii=False, indent=2)
    print("Calibración:", resumen_calib)

    # ---------- 3. Bootstrap CI + delta de Cliff ----------
    rng = np.random.default_rng(42)
    n_test = len(y_test)
    y_test_arr = y_test.to_numpy()
    boot_f1_final, boot_f1_ref, boot_acc_final, boot_acc_ref = [], [], [], []
    for _ in range(200):
        idx = rng.integers(0, n_test, n_test)
        boot_f1_final.append(f1_score(y_test_arr[idx], pred[idx], average="macro", zero_division=0))
        boot_f1_ref.append(f1_score(y_test_arr[idx], pred_ref[idx], average="macro", zero_division=0))
        boot_acc_final.append(accuracy_score(y_test_arr[idx], pred[idx]))
        boot_acc_ref.append(accuracy_score(y_test_arr[idx], pred_ref[idx]))
    boot_f1_final, boot_f1_ref = np.array(boot_f1_final), np.array(boot_f1_ref)

    def ic95(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    def cliffs_delta(x, y_):
        x, y_ = np.asarray(x), np.asarray(y_)
        mayor = sum((xi > y_).sum() for xi in x)
        menor = sum((xi < y_).sum() for xi in x)
        return (mayor - menor) / (len(x) * len(y_))

    delta_cliff_f1 = cliffs_delta(boot_f1_final, boot_f1_ref)
    ic_tabla = pd.DataFrame([
        {"modelo": "XGBoost (finalista)", "métrica": "F1 macro", "media": boot_f1_final.mean(),
         "IC95 inf": ic95(boot_f1_final)[0], "IC95 sup": ic95(boot_f1_final)[1]},
        {"modelo": "Regresión logística (referencia)", "métrica": "F1 macro", "media": boot_f1_ref.mean(),
         "IC95 inf": ic95(boot_f1_ref)[0], "IC95 sup": ic95(boot_f1_ref)[1]},
        {"modelo": "XGBoost (finalista)", "métrica": "Accuracy", "media": np.mean(boot_acc_final),
         "IC95 inf": ic95(boot_acc_final)[0], "IC95 sup": ic95(boot_acc_final)[1]},
        {"modelo": "Regresión logística (referencia)", "métrica": "Accuracy", "media": np.mean(boot_acc_ref),
         "IC95 inf": ic95(boot_acc_ref)[0], "IC95 sup": ic95(boot_acc_ref)[1]},
    ])
    ic_tabla.to_csv(RESULTS3 / "finalista_bootstrap_ic.csv", index=False, encoding="utf-8-sig")
    with open(RESULTS3 / "finalista_tamano_efecto.json", "w", encoding="utf-8") as f:
        json.dump({"delta_cliff_f1_macro": float(delta_cliff_f1),
                   "interpretacion": "grande" if abs(delta_cliff_f1) >= 0.474 else
                                      ("mediano" if abs(delta_cliff_f1) >= 0.33 else
                                       ("pequeño" if abs(delta_cliff_f1) >= 0.147 else "despreciable"))},
                  f, ensure_ascii=False, indent=2)
    print(f"Delta de Cliff (F1 macro, finalista vs referencia): {delta_cliff_f1:.4f}")
    print(ic_tabla)

    # ---------- 4. SHAP local (2 casos) + LIME ----------
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    modelo_entrenado = pipe.named_steps["clf"]
    X_test_t = pipe[:-1].transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()
    nombres_cols = u3.ENG_NUM_COLS + u3.BIN_COLS

    correctos = np.where((pred == y_test.to_numpy()) & (proba.max(axis=1) > 0.95))[0]
    errores = np.where(pred != y_test.to_numpy())[0]
    idx_correcto = int(correctos[0]) if len(correctos) else 0
    idx_error = int(errores[0]) if len(errores) else 1

    explainer = shap.TreeExplainer(modelo_entrenado)
    casos = {}
    for nombre, idx in [("bien_clasificado", idx_correcto), ("mal_clasificado", idx_error)]:
        sv = explainer.shap_values(X_test_t[idx:idx + 1])
        clase_pred = int(pred[idx])
        clase_idx_modelo = clases_modelo_orden.index(clase_pred)
        if isinstance(sv, list):
            contrib = sv[clase_idx_modelo][0]
        else:
            contrib = sv[0, :, clase_idx_modelo] if sv.ndim == 3 else sv[0]
        casos[nombre] = {
            "indice_test": idx, "clase_real": int(y_test.iloc[idx]), "clase_predicha": clase_pred,
            "probabilidad": float(proba[idx, CLASES.index(clase_pred)]),
            "shap_contribuciones": dict(zip(nombres_cols, [float(v) for v in contrib])),
        }

    # LIME sobre los mismos 2 casos, mismo modelo
    ref_muestra = X_test_t[rng.choice(len(X_test_t), size=2000, replace=False)]
    lime_explainer = LimeTabularExplainer(
        ref_muestra, feature_names=nombres_cols, class_names=[str(c) for c in CLASES],
        discretize_continuous=True, random_state=42,
    )

    def predict_fn(arr):
        return modelo_entrenado.predict_proba(arr)

    for nombre, idx in [("bien_clasificado", idx_correcto), ("mal_clasificado", idx_error)]:
        exp = lime_explainer.explain_instance(
            X_test_t[idx], predict_fn, num_features=10,
            labels=(clases_modelo_orden.index(casos[nombre]["clase_predicha"]),),
            num_samples=3000,
        )
        lbl = clases_modelo_orden.index(casos[nombre]["clase_predicha"])
        casos[nombre]["lime_contribuciones"] = {nombres_cols[i]: float(w) for i, w in exp.as_map()[lbl]}
        casos[nombre]["lime_r2"] = float(exp.score)

    with open(RESULTS3 / "shap_casos_locales.json", "w", encoding="utf-8") as f:
        json.dump(casos, f, ensure_ascii=False, indent=2)
    print("SHAP local + LIME guardados para 2 casos.")

    # ---------- 5. Perfilamiento: XGBoost estandar vs optimizado ----------
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline as SkPipeline

    steps_estandar = list(u3.build_preprocessor_steps())
    pipe_estandar = SkPipeline(steps_estandar + [("clf", XGBClassifier(
        objective="multi:softprob", num_class=7, eval_metric="mlogloss",
        tree_method="hist", random_state=42, n_jobs=1,
    ))])

    def perfilar(pipe_, nombre):
        tracemalloc.start()
        t0 = time.time()
        pipe_.fit(X_train, y_train_fit if nombre == "optimizado" else (y_train - 1))
        t_fit = time.time() - t0
        _, pico_fit = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        t0 = time.time()
        pred_ = pipe_.predict(X_test)
        t_pred = time.time() - t0
        _, pico_pred = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        f1_ = f1_score(y_test, pred_ + 1, average="macro", zero_division=0)
        return {"versión": nombre, "tiempo_entrenamiento_s": t_fit, "tiempo_inferencia_s": t_pred,
                "memoria_pico_entrenamiento_MB": pico_fit / 1e6, "memoria_pico_inferencia_MB": pico_pred / 1e6,
                "F1_macro_prueba": f1_}

    perfil_estandar = perfilar(pipe_estandar, "estándar (sin ajustar)")
    perfil_optimizado = perfilar(pipe, "optimizado (finalista)")
    pd.DataFrame([perfil_estandar, perfil_optimizado]).to_csv(
        RESULTS3 / "finalista_perfilamiento.csv", index=False, encoding="utf-8-sig"
    )
    print("Perfilamiento:", perfil_estandar, perfil_optimizado)

    # ---------- 6. Sensibilidad a la semilla (subsample=0.8 para que tenga efecto real) ----------
    hp_semilla = dict(hp_final)
    hp_semilla["subsample"] = 0.8
    hp_semilla["colsample_bytree"] = 0.8
    resultados_semillas = []
    for semilla in [0, 1, 2, 3, 4]:
        params_s = dict(hp_semilla)
        est = XGBClassifier(
            objective="multi:softprob", num_class=7, eval_metric="mlogloss",
            tree_method="hist", random_state=semilla, n_jobs=1, **params_s,
        )
        pipe_s = SkPipeline(list(u3.build_preprocessor_steps()) + [("clf", est)])
        pipe_s.fit(X_train, y_train_fit)
        pred_s = pipe_s.predict(X_test) + 1
        resultados_semillas.append({"semilla": semilla, "F1_macro": f1_score(y_test, pred_s, average="macro", zero_division=0)})
    semillas_df = pd.DataFrame(resultados_semillas)
    semillas_df.to_csv(RESULTS3 / "finalista_sensibilidad_semilla.csv", index=False, encoding="utf-8-sig")
    print("Sensibilidad a la semilla (con subsample=0.8 para introducir aleatoriedad real):")
    print(semillas_df)
    print(f"Media: {semillas_df['F1_macro'].mean():.4f} · DE: {semillas_df['F1_macro'].std():.4f}")

    print("TODO LISTO.")


if __name__ == "__main__":
    main()
