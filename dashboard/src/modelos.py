"""Carga de artefactos y prediccion en vivo (el dashboard nunca entrena ni optimiza).

Los modelos se cargan una sola vez por proceso (functools.lru_cache) desde artifacts/,
donde los dejo scripts/e3_holdout_artefactos.py junto con su metadata.json.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .explicacion import NOMBRES_GRUPOS, kernel_shap
from .preprocesamiento import FEATURES

ARTEFACTOS = Path(__file__).resolve().parents[1] / "artifacts"
BASES = ["KNN", "NB", "LR", "DT", "RF", "XGB", "SVM"]
CLASES = np.arange(1, 8)


@lru_cache(maxsize=None)
def metadata(modelo):
    with open(ARTEFACTOS / f"{modelo}.metadata.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def cargar(modelo):
    return joblib.load(ARTEFACTOS / f"{modelo}.joblib")


def esquema():
    return metadata("RF")["esquema_entrada"]


def _proba_base(modelo, X):
    pipe = cargar(modelo)
    P = pipe.predict_proba(X)
    clases = np.asarray(pipe.classes_) + (1 if modelo == "XGB" else 0)
    salida = np.zeros((len(X), len(CLASES)))
    salida[:, np.searchsorted(CLASES, clases)] = P
    return salida


TAM_LOTE = 64  # filas por lote: acota la memoria pico (KernelSHAP evalua miles de filas a la vez)


def predecir_proba(modelo, X):
    """Probabilidades (n, 7) de un modelo base o del ECLC para un DataFrame con las 54 columnas."""
    X = X[FEATURES]
    if len(X) > TAM_LOTE:
        return np.vstack([predecir_proba(modelo, X.iloc[i:i + TAM_LOTE]) for i in range(0, len(X), TAM_LOTE)])
    if modelo != "ECLC":
        return _proba_base(modelo, X)
    eclc = cargar("ECLC")
    P = np.stack([_proba_base(m, X) for m in metadata("ECLC")["modelos_base"]])
    return eclc.combinar(P, X=X)


def pesos_eclc(X):
    """Pesos w_mc(x) del ECLC (n, M, C): cuanto confia en cada modelo para cada clase."""
    return cargar("ECLC").pesos(X=X[FEATURES])


def fila_desde_formulario(valores, area, suelo):
    """Arma la fila de 54 columnas a partir de las 10 numericas, el area y el tipo de suelo."""
    fila = {c: 0 for c in FEATURES}
    fila.update({k: float(v) for k, v in valores.items()})
    fila[f"Wilderness_Area_{int(area)}"] = 1
    fila[f"Soil_Type_{int(suelo)}"] = 1
    return pd.DataFrame([fila])[FEATURES]


def validar(valores):
    """Lista de mensajes de error si algun valor esta vacio o fuera del rango observado."""
    rangos = {e["nombre"]: (e["min"], e["max"]) for e in esquema()}
    errores = []
    for k, v in valores.items():
        if v is None:
            errores.append(f"Falta el valor de {k}.")
        elif not (rangos[k][0] <= v <= rangos[k][1]):
            errores.append(f"{k} = {v} esta fuera del rango observado [{rangos[k][0]}, {rangos[k][1]}].")
    return errores


def explicar(modelo, X, fondo, clase, n_coaliciones=256):
    """KernelSHAP agrupado de la probabilidad de 'clase' (1..7) para la fila X."""
    phi, base, fx = kernel_shap(lambda D: predecir_proba(modelo, D), X, fondo, clase - 1,
                                n_coaliciones=n_coaliciones)
    return pd.DataFrame({"variable": NOMBRES_GRUPOS, "shap": phi}), base, fx
