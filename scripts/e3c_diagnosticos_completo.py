"""Diagnósticos de la prueba reservada con los modelos entrenados con el conjunto completo.

El dashboard carga modelos entrenados con la submuestra de 40.000 celdas (memoria de Render). Este script
precalcula, para los modelos del protocolo de datos completos (464.809 celdas), los mismos diagnósticos
livianos que muestra la pestaña Diagnóstico: confusión, ROC, precisión-recall, calibración, Brier y F1.

Las probabilidades de los 7 modelos base y de HistGradientBoosting están en resultados/e3/completo/holdout.npz;
las del ECLC se recalculan con los hiperparámetros elegidos por TPE en e3c_meta.py --holdout y se verifica que
reproducen exactamente sus predicciones guardadas (pred_holdout.npz).

Salidas: resultados/e3/completo/holdout_probas_eclc.npz y dashboard/datos/e3/diagnosticos_completo.json
Uso: python scripts/e3c_diagnosticos_completo.py
"""
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import _e3 as e3
import _utils3 as u3
import e3c_meta as cm
from e3_preparar_dashboard import DEST, diagnostico_de

SALIDA = e3.RES_E3 / "completo"


def probas_eclc():
    ruta = SALIDA / "holdout_probas_eclc.npz"
    if ruta.exists():
        return np.load(ruta)["P"]
    d = np.load(SALIDA / "holdout.npz")
    hp = json.load(open(SALIDA / "meta_holdout.json", encoding="utf-8"))["hiperparametros"]
    _, _, X_test, _, X_train, _ = u3.get_data(n=u3.N_BENCHMARK)
    modelo = cm.eclc(**hp).ajustar_desde_oof(X_train, d["y_tr"], d["P_oof"].astype(np.float64))
    Zte = modelo.transformador_vecinos_.transform(X_test).astype(np.float32)
    t0 = time.perf_counter()
    P = modelo.combinar(d["P_te"].astype(np.float64), Z=Zte)
    print(f"ECLC con hp={hp}: {time.perf_counter() - t0:.0f}s", flush=True)
    guardado = np.load(SALIDA / "pred_holdout.npz")["ECLC"]
    coincide = float((e3.CLASES[P.argmax(1)] == guardado).mean())
    print(f"coincidencia con las predicciones guardadas: {coincide:.6f}", flush=True)
    assert coincide > 0.999, "las probabilidades recalculadas no reproducen el ECLC del protocolo completo"
    np.savez_compressed(ruta, P=P.astype(np.float32))
    return P


def main():
    d = np.load(SALIDA / "holdout.npz")
    y = d["y_te"]
    probas = {str(m): d["P_te"][i].astype(float) for i, m in enumerate(d["modelos"])}
    probas["HGB"] = d["P_hgb"].astype(float)
    probas["ECLC"] = probas_eclc().astype(float)
    salida = {}
    for m, P in probas.items():
        P = P / P.sum(1, keepdims=True)
        pred = e3.CLASES[P.argmax(1)]
        salida[m] = {**diagnostico_de(y, P), "f1_macro": float(f1_score(y, pred, average="macro")),
                     "accuracy": float(accuracy_score(y, pred))}
        print(f"{m:5s} F1 macro {salida[m]['f1_macro']:.4f}  accuracy {salida[m]['accuracy']:.4f}  "
              f"Brier {salida[m]['brier']:.3f}  ECE {salida[m]['ece']:.3f}", flush=True)
    json.dump(salida, open(DEST / "diagnosticos_completo.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
