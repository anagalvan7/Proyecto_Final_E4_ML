"""KernelSHAP agrupado (Lundberg y Lee, 2017) implementado con numpy.

Jugadores: las 10 variables numericas originales, el area silvestre (4 indicadoras) y el
tipo de suelo (40 indicadoras). Agrupar las indicadoras evita repartir la contribucion de
una sola categoria entre 40 columnas que no pueden variar por separado.

v(S) = promedio sobre el fondo B de f(x_S, b_{-S})[c]. Los valores de Shapley se estiman
por minimos cuadrados ponderados con el nucleo de Shapley
    pi(S) = (G - 1) / (C(G, |S|) |S| (G - |S|))
y la restriccion de eficiencia sum_i phi_i = f(x) - E_B f(b). Con G = 12 hay 4.094
coaliciones no triviales; si n_coaliciones >= 4.094 se enumeran todas (resultado exacto),
si no, se muestrean por tamano con muestreo pareado (S y su complemento).
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .preprocesamiento import NUM_COLS, SOIL_COLS, WILD_COLS

GRUPOS = [[c] for c in NUM_COLS] + [WILD_COLS, SOIL_COLS]
NOMBRES_GRUPOS = NUM_COLS + ["Wilderness_Area", "Soil_Type"]


def _coaliciones(G, n, rng):
    """Matriz binaria de coaliciones (sin la vacia ni la completa)."""
    total = 2 ** G - 2
    if n >= total:
        filas = [np.isin(np.arange(G), c).astype(float)
                 for s in range(1, G) for c in combinations(range(G), s)]
        return np.array(filas)
    tamanos = np.arange(1, G)
    peso = (G - 1) / (tamanos * (G - tamanos))
    peso = peso / peso.sum()
    filas = []
    while len(filas) < n:
        s = rng.choice(tamanos, p=peso)
        z = np.zeros(G)
        z[rng.choice(G, size=s, replace=False)] = 1
        filas.append(z)
        filas.append(1 - z)
    return np.array(filas[:n])


def kernel_shap(f, x, fondo, clase, n_coaliciones=512, semilla=0, grupos=GRUPOS):
    """Valores de Shapley agrupados de f(x)[clase].

    f : callable DataFrame -> probabilidades (n, C)
    x : DataFrame de una fila con las 54 columnas originales
    fondo : DataFrame de referencia (pocas filas: el costo es n_coaliciones * len(fondo))
    Devuelve (phi, valor_base, f_x).
    """
    rng = np.random.default_rng(semilla)
    G = len(grupos)
    Z = _coaliciones(G, n_coaliciones, rng)
    xv = x.iloc[0]
    bloques = []
    for z in Z:
        B = fondo.copy()
        for g in np.where(z == 1)[0]:
            for col in grupos[g]:
                B[col] = xv[col]
        bloques.append(B)
    todo = pd.concat([x, fondo] + bloques, ignore_index=True)
    P = f(todo)[:, clase]
    f_x = P[0]
    nb = len(fondo)
    base = P[1:1 + nb].mean()
    v = P[1 + nb:].reshape(len(Z), nb).mean(axis=1)
    tam = Z.sum(axis=1)
    pi = (G - 1) / (np.array([comb(G, int(s)) for s in tam]) * tam * (G - tam))
    # Eliminacion de la restriccion: phi_G = (f_x - base) - sum_{i<G} phi_i
    delta = f_x - base
    A = Z[:, :-1] - Z[:, [-1]]
    b = (v - base) - Z[:, -1] * delta
    Wa = A * pi[:, None]
    phi_rest = np.linalg.lstsq(A.T @ Wa, Wa.T @ b, rcond=None)[0]
    phi = np.append(phi_rest, delta - phi_rest.sum())
    return phi, float(base), float(f_x)
