"""Utilidades compartidas por los experimentos del Entregable 3.

- Seleccion de la mejor combinacion de cada modelo base (seccion 4.1 de la guia).
- Particiones externas repetidas: la repeticion 0 es exactamente la particion del
  benchmark (StratifiedKFold(3, shuffle=True, random_state=42)); las repeticiones r > 0
  usan random_state = 42 + r.
- Ajuste de un modelo base con su balanceo y sus hiperparametros, devolviendo
  probabilidades con las columnas en el orden de las clases 1..7.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
RES_E3 = RES / "e3"
RES_E3.mkdir(exist_ok=True)
sys.path.insert(0, str(RAIZ / "dashboard"))

MODELOS_ORDEN = ["KNN", "NB", "LR", "DT", "RF", "XGB", "SVM"]
CLASES = np.arange(1, 8)
N_REPETICIONES = 10
N_PLIEGUES = 3
N_SVM_MAX = 6_000          # misma escala de entrenamiento del SVM en su benchmark (9.000 filas)
ORDEN_OPT = {"grid": 0, "random": 1, "bayes": 2, "genetico": 3}


def tabla_maestra():
    t = pd.read_csv(RES / "tabla_maestra.csv", encoding="utf-8-sig")
    return t[t["aplicable"]].copy()


def mejores_combinaciones(tabla=None):
    """Mejor combinacion de cada modelo segun el F1 macro medio del bucle externo.

    Desempates (frecuentes: varios optimizadores llegan a la misma configuracion):
    menor desviacion estandar entre pliegues y, despues, menor tiempo total.
    """
    t = tabla_maestra() if tabla is None else tabla
    t = t.assign(_r=-t["f1_macro_media"].round(6), _sd=t["f1_macro_sd"].round(6))
    t = t.sort_values(["modelo", "_r", "_sd", "tiempo_total_s"])
    mejores = t.groupby("modelo", sort=False).head(1).drop(columns=["_r", "_sd"])
    mejores = mejores.set_index("modelo").loc[MODELOS_ORDEN].reset_index()
    mejores["params_pliegues"] = mejores["mejores_hiperparametros_por_pliegue"].apply(json.loads)
    mejores["params_moda"] = mejores["params_pliegues"].apply(params_moda)
    return mejores


def params_moda(lista):
    """Configuracion mas frecuente entre los pliegues externos (empate: la del primero)."""
    claves = [json.dumps(p, sort_keys=True) for p in lista]
    conteo = Counter(claves)
    mejor = max(claves, key=lambda c: (conteo[c], -claves.index(c)))
    return json.loads(mejor)


def particion(rep, X, y):
    """Lista de (idx_entrenamiento, idx_prueba) de la repeticion rep."""
    cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=u3.RANDOM_STATE + rep)
    return list(cv.split(X, y))


def params_para(fila, rep, pliegue):
    """Repeticion 0: los hiperparametros que eligio el bucle interno del benchmark en ese
    pliegue. Repeticiones r > 0: la configuracion modal del benchmark."""
    return fila["params_pliegues"][pliegue] if rep == 0 else fila["params_moda"]


def _submuestra(X, y, n, semilla=u3.RANDOM_STATE):
    if len(X) <= n:
        return X, y
    return u3.stratified_sample(X, y, n, random_state=semilla)


def ajustar_modelo(modelo, balanceo, params, X, y):
    """Ajusta el pipeline (preprocesamiento + balanceo + modelo); y en 1..7."""
    params = dict(params)
    if modelo == "SVM":
        params["probability"] = True
        X, y = _submuestra(X, y, N_SVM_MAX)
    y_fit = (y - 1) if modelo == "XGB" else y
    fit_params = u3._fit_params_for(modelo, balanceo, y_fit)
    pipe, _ = u3.robust_fit(modelo, balanceo, params, X, y_fit, fit_params)
    return pipe


def proba_ordenada(pipe, modelo, X):
    """predict_proba con columnas en el orden 1..7 (rellena con 0 una clase ausente)."""
    P = pipe.predict_proba(X)
    clases = np.asarray(pipe.classes_) + (1 if modelo == "XGB" else 0)
    salida = np.zeros((len(X), len(CLASES)))
    salida[:, np.searchsorted(CLASES, clases)] = P
    return salida


def probas_oof(modelo, balanceo, params, X, y, semilla=u3.RANDOM_STATE):
    """Probabilidades fuera de pliegue (3 pliegues estratificados) sobre X."""
    P = np.zeros((len(X), len(CLASES)))
    cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=semilla)
    for tr, va in cv.split(X, y):
        pipe = ajustar_modelo(modelo, balanceo, params, X.iloc[tr], y.iloc[tr])
        P[va] = proba_ordenada(pipe, modelo, X.iloc[va])
    return P


def espacio_vecinos(X_ajuste):
    """Preprocesamiento del benchmark (ingenieria + escalado) ajustado en X_ajuste: define
    el espacio donde el ECLC busca vecinos."""
    from sklearn.pipeline import Pipeline
    return Pipeline(u3.build_preprocessor_steps()).fit(X_ajuste)
