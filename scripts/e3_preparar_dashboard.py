"""Genera dashboard/datos/e3/: archivos livianos con todo lo que muestra la seccion Modelos.

Uso:
    python scripts/e3_preparar_dashboard.py --shap MODELO   (SHAP global de un modelo; en paralelo)
    python scripts/e3_preparar_dashboard.py                 (todo lo demas y une los SHAP)
"""
import argparse
import glob
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, \
    recall_score, roc_curve

import _e3 as e3
import _utils3 as u3
from src import preprocesamiento
from src.explicacion import NOMBRES_GRUPOS, kernel_shap

u3.engineer_features = preprocesamiento.engineer_features
DEST = e3.RAIZ / "dashboard" / "datos" / "e3"
DEST.mkdir(parents=True, exist_ok=True)
MODELOS = e3.MODELOS_ORDEN + ["ECLC"]
N_SHAP = 60


def _curva(x, y, n=60):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [round(float(v), 4) for v in x[idx]], [round(float(v), 4) for v in y[idx]]


def diagnosticos():
    h = np.load(e3.RES_E3 / "holdout_probas.npz")
    y = h["y_test"]
    salida = {}
    for m in MODELOS:
        P = h[m].astype(float)
        P = P / P.sum(1, keepdims=True)
        salida[m] = diagnostico_de(y, P)
    json.dump(salida, open(DEST / "diagnosticos.json", "w", encoding="utf-8"))


def diagnostico_de(y, P):
    """Confusion, ROC, precision-recall, calibracion, ECE, Brier y metricas por clase de unas probabilidades (n, 7)."""
    pred = e3.CLASES[P.argmax(1)]
    cm = confusion_matrix(y, pred, labels=e3.CLASES)
    roc, pr = {}, {}
    for i, c in enumerate(e3.CLASES):
        yb = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(yb, P[:, i])
        pre, rec, _ = precision_recall_curve(yb, P[:, i])
        roc[int(c)] = {"fpr": _curva(fpr, tpr)[0], "tpr": _curva(fpr, tpr)[1], "auc": float(auc(fpr, tpr))}
        pr[int(c)] = {"recall": _curva(rec[::-1], pre[::-1])[0], "precision": _curva(rec[::-1], pre[::-1])[1],
                      "ap": float(average_precision_score(yb, P[:, i]))}
    conf = P.max(1)
    acierto = (pred == y)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    calib = [{"confianza": float(conf[idx == b].mean()), "exactitud": float(acierto[idx == b].mean()),
              "n": int((idx == b).sum())} for b in range(10) if (idx == b).sum() > 0]
    ece = sum(r["n"] * abs(r["confianza"] - r["exactitud"]) for r in calib) / len(y)
    Y = np.eye(7)[np.searchsorted(e3.CLASES, y)]
    return {"confusion": cm.tolist(), "roc": roc, "pr": pr, "calibracion": calib, "ece": float(ece),
            "brier": float(((P - Y) ** 2).sum(1).mean()),
            "f1_clase": f1_score(y, pred, labels=e3.CLASES, average=None).tolist(),
            "precision_clase": precision_score(y, pred, labels=e3.CLASES, average=None).tolist(),
            "recall_clase": recall_score(y, pred, labels=e3.CLASES, average=None).tolist()}


def celdas_shap():
    _, _, X_test, y_test, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    Xs, ys = u3.stratified_sample(X_test, y_test, N_SHAP, random_state=7)
    return Xs, ys


def fondo():
    X, y, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    Xf, _ = u3.stratified_sample(X, y, 10, random_state=3)
    return Xf


def shap_modelo(modelo):
    """phi de la clase predicha para N_SHAP celdas de la prueba reservada."""
    import joblib
    from src.modelos import predecir_proba
    Xs, ys = celdas_shap()
    Xf = fondo()
    filas = []
    for i in range(len(Xs)):
        x = Xs.iloc[[i]]
        p = predecir_proba(modelo, x)[0]
        c = int(p.argmax())
        phi, base, fx = kernel_shap(lambda D: predecir_proba(modelo, D), x, Xf, c, n_coaliciones=256)
        filas.append({"celda": i, "clase_real": int(ys.iloc[i]), "clase_pred": c + 1, "base": base, "fx": fx,
                      **{g: float(v) for g, v in zip(NOMBRES_GRUPOS, phi)},
                      **{f"valor_{g}": float(x.iloc[0][g]) for g in preprocesamiento.NUM_COLS}})
    pd.DataFrame(filas).to_csv(e3.RES_E3 / f"shap_{modelo}.csv", index=False)
    print(modelo, "SHAP listo", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap")
    args = ap.parse_args()
    if args.shap:
        shap_modelo(args.shap)
        return
    for nombre in ["holdout_resumen.json", "estadistica_veredicto.json", "estadistica_resumen.csv",
                   "estadistica_pareadas.csv", "estadistica_ablacion.csv", "datasets_adicionales.csv",
                   "robustez.csv", "robustez_errores.csv", "complejidad_meta.csv"] + \
            [f"holdout_confusion_{m}.csv" for m in MODELOS + ["HGB"]]:
        if (e3.RES_E3 / nombre).exists():
            shutil.copy2(e3.RES_E3 / nombre, DEST / nombre)
    diagnosticos()
    fondo().to_csv(DEST / "fondo_shap.csv", index=False)
    Xs, ys = celdas_shap()
    ej = Xs.copy()
    ej["Cover_Type"] = ys.to_numpy()
    ej.reset_index(drop=True).to_csv(DEST / "ejemplos.csv", index=False)
    # convergencia de la optimizacion bayesiana del ECLC en los 30 pliegues
    conv = []
    for ruta in sorted(glob.glob(str(e3.RES_E3 / "meta" / "rep*_f*.json"))):
        d = json.load(open(ruta, encoding="utf-8"))
        mejor = -1
        for t, h in enumerate(d["historia_optuna"], 1):
            mejor = max(mejor, h["f1_cv"])
            conv.append({"rep": d["rep"], "pliegue": d["pliegue"], "ensayo": t, "f1_cv": h["f1_cv"],
                         "mejor_hasta_ahora": mejor, "k": h["k"], "beta": h["beta"], "tau": h["tau"]})
    pd.DataFrame(conv).to_csv(DEST / "convergencia_optuna.csv", index=False, encoding="utf-8-sig")
    # SHAP global
    globales = {}
    for m in MODELOS:
        ruta = e3.RES_E3 / f"shap_{m}.csv"
        if ruta.exists():
            s = pd.read_csv(ruta)
            globales[m] = {"importancia": {g: float(s[g].abs().mean()) for g in NOMBRES_GRUPOS},
                           "puntos": s.to_dict(orient="list")}
    json.dump(globales, open(DEST / "shap_global.json", "w", encoding="utf-8"))
    for c in ["resultados/tabla_maestra.csv", "resultados/friedman_resumen.json"]:
        shutil.copy2(e3.RAIZ / c, e3.RAIZ / "dashboard" / "datos" / "resultados" / c.split("/")[-1])
    print("dashboard/datos/e3 listo")


if __name__ == "__main__":
    main()
