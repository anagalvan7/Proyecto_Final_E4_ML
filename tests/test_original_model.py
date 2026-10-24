"""Pruebas unitarias del ensamble por competencia local y por clase (ECLC).

Cubren los cuatro requisitos de la seccion 5.4 de la guia del Entregable 3:
(i) aprende en un problema sintetico con solucion conocida, (ii) reproducibilidad con la
misma semilla, (iii) manejo de entradas invalidas, (iv) compatibilidad con
sklearn.base.clone; y ademas verifican numericamente las Proposiciones 1 a 3 del capitulo 5.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from src.original_model import (  # noqa: E402
    EnsambleCompetenciaLocal, perdidas_binarias, pesos_gibbs, vecinos_mas_cercanos,
)


class ExpertoRegional(ClassifierMixin, BaseEstimator):
    """Experto sintetico: acierta con probabilidad 0,95 si x0 tiene el signo indicado y
    responde al azar (probabilidad 0,5) en la otra mitad del espacio."""

    def __init__(self, signo=1):
        self.signo = signo

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        verdad = (X[:, 1] > 0).astype(int)          # la etiqueta real depende de x1
        region = np.sign(X[:, 0]) == self.signo
        p1 = np.where(region, np.where(verdad == 1, 0.95, 0.05), 0.5)
        return np.column_stack([1 - p1, p1])


def datos_regionales(n=1200, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 1] > 0).astype(int)
    return X, y


def base_iris():
    return [
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
        ("nb", GaussianNB()),
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ]


# ---------------------------------------------------------------- (i) aprende
def test_aprende_solucion_conocida():
    """Cada experto acierta solo en su mitad del plano; el ensamble debe elegir el experto
    correcto en cada region y superar ampliamente a ambos."""
    X, y = datos_regionales()
    Xt, yt = datos_regionales(semilla=1)
    expertos = [("izq", ExpertoRegional(-1)), ("der", ExpertoRegional(1))]
    modelo = EnsambleCompetenciaLocal(expertos, k=30, beta=10.0, tau=1.0).fit(X, y)
    acc_ens = (modelo.predict(Xt) == yt).mean()
    acc_expertos = [((e.predict_proba(Xt)[:, 1] > 0.5).astype(int) == yt).mean() for _, e in expertos]
    assert acc_ens > 0.97
    assert acc_ens > max(acc_expertos) + 0.2
    W = modelo.pesos(Xt)                     # (n, M, C)
    der = Xt[:, 0] > 0.2
    izq = Xt[:, 0] < -0.2
    assert W[der, 1, :].mean() > 0.9         # a la derecha domina el experto "der"
    assert W[izq, 0, :].mean() > 0.9


def test_supera_o_iguala_al_promedio_en_iris():
    X, y = load_iris(return_X_y=True)
    eclc = EnsambleCompetenciaLocal(base_iris(), k=15, beta=5.0, tau=5.0, random_state=0)
    s = cross_val_score(eclc, X, y, cv=5, scoring="f1_macro")
    assert s.mean() > 0.9


# ---------------------------------------------------------------- (ii) reproducibilidad
def test_reproducible_con_la_misma_semilla():
    X, y = load_iris(return_X_y=True)
    p1 = EnsambleCompetenciaLocal(base_iris(), random_state=7).fit(X, y).predict_proba(X)
    p2 = EnsambleCompetenciaLocal(base_iris(), random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_probabilidades_validas():
    X, y = load_iris(return_X_y=True)
    P = EnsambleCompetenciaLocal(base_iris()).fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------- (iii) entradas invalidas
@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 2.5}, {"beta": -1.0}, {"beta": np.inf},
                                    {"tau": -0.1}, {"n_pliegues": 1}, {"estimadores": []}])
def test_hiperparametros_invalidos(kwargs):
    X, y = load_iris(return_X_y=True)
    params = {"estimadores": base_iris(), **kwargs}
    with pytest.raises(ValueError):
        EnsambleCompetenciaLocal(**params).fit(X, y)


def test_entradas_invalidas():
    X, y = load_iris(return_X_y=True)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        EnsambleCompetenciaLocal(base_iris()).fit(X_nan, y)
    with pytest.raises(ValueError):
        EnsambleCompetenciaLocal(base_iris()).fit(X, np.zeros(len(y)))   # una sola clase
    modelo = EnsambleCompetenciaLocal(base_iris()).fit(X, y)
    with pytest.raises(ValueError):
        modelo.predict(X[:, :3])                                         # columnas distintas


def test_no_ajustado():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        EnsambleCompetenciaLocal(base_iris()).predict(np.zeros((2, 4)))


# ---------------------------------------------------------------- (iv) clone
def test_reconocido_como_clasificador():
    from sklearn.base import is_classifier
    assert is_classifier(EnsambleCompetenciaLocal(base_iris()))


def test_compatibilidad_clone():
    modelo = EnsambleCompetenciaLocal(base_iris(), k=11, beta=3.0, tau=2.0)
    copia = clone(modelo)
    assert copia.get_params()["k"] == 11 and copia.get_params()["beta"] == 3.0
    assert copia is not modelo
    X, y = load_iris(return_X_y=True)
    copia.fit(X, y)
    assert not hasattr(modelo, "estimadores_")      # clonar no comparte estado ajustado


# ---------------------------------------------------------------- propiedades teoricas
def test_prop1_pesos_minimizan_objetivo_regularizado():
    """Proposicion 1: w = softmax(-beta L) minimiza <w, L> + (1/beta) sum w log w en el simplex."""
    rng = np.random.default_rng(3)
    L = rng.uniform(0, 2, size=6)
    beta = 2.5
    w = pesos_gibbs(L[None, :], beta, eje=1)[0]

    def F(v):
        return v @ L + (1 / beta) * np.sum(v * np.log(np.maximum(v, 1e-300)))

    for _ in range(2000):
        v = rng.dirichlet(np.ones(6))
        assert F(w) <= F(v) + 1e-12


def test_prop2_cota_de_jensen_y_gibbs():
    """Proposicion 2: en la vecindad de x, la perdida binaria del ensamble con los pesos de x
    no supera el promedio ponderado de las perdidas ni min_m L_mc + log(M)/beta."""
    rng = np.random.default_rng(5)
    M, k, C = 5, 40, 4
    P = rng.dirichlet(np.ones(C), size=(M, k))           # (M, k, C)
    y = rng.integers(0, C, size=k)
    a = rng.uniform(0.1, 1, size=k)
    l = perdidas_binarias(P, y, C)                      # (k, M, C)
    L = np.einsum("j,jmc->mc", a, l) / a.sum()           # (M, C)
    for beta in (0.5, 3.0, 20.0):
        W = pesos_gibbs(L, beta, eje=0)                  # (M, C)
        q = np.einsum("mc,mjc->jc", W, P)                # (k, C)
        l_ens = perdidas_binarias(q[None], y, C)[:, 0, :]
        L_ens = (a @ l_ens) / a.sum()                    # (C,)
        prom = (W * L).sum(axis=0)
        assert np.all(L_ens <= prom + 1e-9)
        assert np.all(prom <= L.min(axis=0) + np.log(M) / beta + 1e-9)


def test_casos_limite_beta_cero_y_tau_infinito():
    """beta = 0 reproduce el promedio simple; tau = inf da pesos iguales para todas las filas."""
    X, y = load_iris(return_X_y=True)
    m0 = EnsambleCompetenciaLocal(base_iris(), beta=0.0).fit(X, y)
    P = m0.probas_base(X)
    prom = P.mean(axis=0)
    np.testing.assert_allclose(m0.predict_proba(X), prom / prom.sum(axis=1, keepdims=True), atol=1e-12)
    m_inf = EnsambleCompetenciaLocal(base_iris(), tau=np.inf).fit(X, y)
    W = m_inf.pesos(X)
    np.testing.assert_allclose(W, np.broadcast_to(W[0], W.shape))


def test_busqueda_de_vecinos_exacta():
    rng = np.random.default_rng(11)
    R = rng.normal(size=(500, 6)).astype(np.float32)
    Q = rng.normal(size=(37, 6)).astype(np.float32)
    idx, dist = vecinos_mas_cercanos(R, Q, k=9, tam_bloque=10)
    d_full = np.linalg.norm(Q[:, None, :] - R[None, :, :], axis=2)
    esperado = np.sort(d_full, axis=1)[:, :9]
    np.testing.assert_allclose(dist, esperado, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.take_along_axis(d_full, idx, axis=1), esperado, rtol=1e-4, atol=1e-4)


def test_vecinos_precalculados_dan_el_mismo_resultado():
    """Sin empates de distancia, reutilizar una busqueda mas amplia da exactamente el mismo resultado.
    (Iris tiene filas duplicadas; con empates en el k-esimo vecino el orden puede variar.)"""
    X, y = load_iris(return_X_y=True)
    X = X + np.random.default_rng(0).normal(scale=1e-3, size=X.shape)
    m = EnsambleCompetenciaLocal(base_iris(), k=7, beta=4.0, tau=3.0).fit(X, y)
    P = m.probas_base(X)
    Z = m._espacio(X)
    vec = vecinos_mas_cercanos(m.Z_ref_, Z, 30)
    np.testing.assert_allclose(m.combinar(P, X=X), m.combinar(P, vecinos=vec), atol=1e-12)
    with pytest.raises(ValueError):
        EnsambleCompetenciaLocal(base_iris(), k=40).fit(X, y).combinar(P, vecinos=vec)
