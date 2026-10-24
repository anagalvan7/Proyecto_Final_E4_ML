"""Combinador multinomial regional con contraccion jerarquica (CRCJ).

Segundo modelo original del Entregable 3. Combina las probabilidades de M modelos base
con una regresion logistica multinomial propia cuyos coeficientes dependen de la region
del espacio de variables a la que pertenece la observacion, contraidos hacia un
combinador global:

    f(x) = [log p_11(x), ..., log p_MC(x), 1]                (caracteristicas, F = M*C + 1)
    r(x) = region de x (k-medias propio en el espacio preprocesado)
    p(c | x) = softmax_c( Theta_{r(x)} f(x) )

    min_{Theta_0, Theta_1..R}  sum_i s_i * CE(y_i, softmax(Theta_{r_i} f_i))
                               + (lam0 / 2) ||Theta_0||^2
                               + (lam  / 2) sum_r n_r/n * ||Theta_r - Theta_0||^2

s_i = (n / (C n_{y_i}))^gamma pondera las clases (gamma = 0: sin ponderar; 1: balanceado).
Con lam0, lam > 0 el objetivo es estrictamente convexo: el optimo es unico (Proposicion 4)
y se encuentra con L-BFGS (scipy.optimize). lam -> inf recupera el stacking global;
R = 1 tambien.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

EPS = 1e-6


def kmedias(Z, R, semilla=0, iteraciones=50, tol=1e-6):
    """k-medias de Lloyd con inicializacion k-means++ (numpy). Devuelve los centroides.

    Cada iteracion no aumenta la suma de distancias cuadradas (asignacion y actualizacion
    son minimizaciones exactas por coordenadas), por lo que termina en un numero finito de
    pasos o al alcanzar 'iteraciones'.
    """
    rng = np.random.default_rng(semilla)
    n = len(Z)
    C = [Z[rng.integers(n)]]
    d2 = ((Z - C[0]) ** 2).sum(1)
    for _ in range(1, R):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        C.append(Z[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((Z - C[-1]) ** 2).sum(1))
    C = np.array(C)
    previo = np.inf
    for _ in range(iteraciones):
        etiqueta, dist = asignar(Z, C)
        costo = dist.sum()
        for r in range(R):
            m = etiqueta == r
            if m.any():
                C[r] = Z[m].mean(0)
        if previo - costo <= tol * max(previo, 1.0):
            break
        previo = costo
    return C


def asignar(Z, C):
    d2 = (Z ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * Z @ C.T
    etiqueta = d2.argmin(1)
    return etiqueta, np.maximum(d2[np.arange(len(Z)), etiqueta], 0)


def caracteristicas(P):
    """P (M, n, C) -> f (n, M*C + 1): log-probabilidades y termino independiente."""
    M, n, C = P.shape
    L = np.log(np.clip(P, EPS, 1.0)).transpose(1, 0, 2).reshape(n, M * C)
    return np.hstack([L, np.ones((n, 1))])


class CombinadorRegional(ClassifierMixin, BaseEstimator):
    """Combinador multinomial regional con contraccion jerarquica (CRCJ).

    Parameters
    ----------
    estimadores : list of (str, estimator)
        Modelos base con predict_proba (pueden venir de librerias).
    n_regiones : int
        Numero de regiones R (k-medias). R = 1 es el stacking multinomial global.
    lam : float > 0
        Fuerza de la contraccion de cada region hacia el combinador global.
    lam0 : float > 0
        Regularizacion L2 del combinador global.
    gamma : float en [0, 1]
        Exponente del peso de clase s_c = (n / (C n_c))^gamma.
    n_pliegues : int
        Pliegues para las probabilidades fuera de pliegue.
    transformador_regiones : transformer o None
        Espacio donde se definen las regiones (None: estandarizacion de las columnas).
    random_state : int
    """

    def __init__(self, estimadores=None, n_regiones=8, lam=10.0, lam0=1.0, gamma=0.5, n_pliegues=3,
                 transformador_regiones=None, random_state=42, max_iter=500):
        self.estimadores = estimadores
        self.n_regiones = n_regiones
        self.lam = lam
        self.lam0 = lam0
        self.gamma = gamma
        self.n_pliegues = n_pliegues
        self.transformador_regiones = transformador_regiones
        self.random_state = random_state
        self.max_iter = max_iter

    # ------------------------------------------------------------------ validacion
    def _validar(self):
        if not self.estimadores:
            raise ValueError("Se requiere al menos un estimador base.")
        if not isinstance(self.n_regiones, numbers.Integral) or self.n_regiones < 1:
            raise ValueError(f"n_regiones debe ser un entero >= 1; se recibio {self.n_regiones!r}.")
        for nombre in ("lam", "lam0"):
            v = getattr(self, nombre)
            if not isinstance(v, numbers.Real) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{nombre} debe ser un real finito > 0; se recibio {v!r}.")
        if not isinstance(self.gamma, numbers.Real) or not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma debe estar en [0, 1]; se recibio {self.gamma!r}.")
        if not isinstance(self.n_pliegues, numbers.Integral) or self.n_pliegues < 2:
            raise ValueError("n_pliegues debe ser un entero >= 2.")

    def _espacio(self, X, ajustar=False):
        if self.transformador_regiones is not None:
            if ajustar:
                self.transformador_regiones_ = clone(self.transformador_regiones).fit(X)
            Z = np.asarray(self.transformador_regiones_.transform(X), dtype=np.float64)
        else:
            Z = check_array(X, dtype=np.float64)
            if ajustar:
                self.media_ = Z.mean(0)
                self.escala_ = Z.std(0)
                self.escala_[self.escala_ < 1e-12] = 1.0
            Z = (Z - self.media_) / self.escala_
        return Z

    def _proba_base(self, est, X):
        P = est.predict_proba(X)
        clases_est = getattr(est, "classes_", None)
        if clases_est is None or len(clases_est) == len(self.classes_):
            return P
        salida = np.zeros((P.shape[0], len(self.classes_)))
        salida[:, np.asarray(clases_est, dtype=int)] = P
        return salida

    # ------------------------------------------------------------------ ajuste
    def fit(self, X, y):
        self._validar()
        Xv, y = check_X_y(X, y, dtype=np.float64)
        X_base = X if hasattr(X, "iloc") else Xv
        self.classes_, y_cod = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Se requieren al menos dos clases.")
        n, C, M = len(y_cod), len(self.classes_), len(self.estimadores)
        P_oof = np.zeros((M, n, C))
        cv = StratifiedKFold(self.n_pliegues, shuffle=True, random_state=self.random_state)
        for tr, va in cv.split(Xv, y_cod):
            X_tr = X_base.iloc[tr] if hasattr(X_base, "iloc") else X_base[tr]
            X_va = X_base.iloc[va] if hasattr(X_base, "iloc") else X_base[va]
            for m, (_, est) in enumerate(self.estimadores):
                P_oof[m, va] = self._proba_base(clone(est).fit(X_tr, y_cod[tr]), X_va)
        self.estimadores_ = [(nombre, clone(est).fit(X_base, y_cod)) for nombre, est in self.estimadores]
        return self.ajustar_desde_oof(X_base, y, P_oof)

    def ajustar_desde_oof(self, X, y, P_oof):
        """Ajusta regiones y combinadores a partir de probabilidades fuera de pliegue (M, n, C)."""
        self._validar()
        y = np.asarray(y)
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(y)
        y_cod = np.searchsorted(self.classes_, y)
        C = len(self.classes_)
        F = caracteristicas(np.asarray(P_oof, dtype=np.float64))
        n, d = F.shape
        Z = self._espacio(X, ajustar=True)
        R = int(self.n_regiones)
        self.centroides_ = kmedias(Z, R, semilla=self.random_state) if R > 1 else Z.mean(0, keepdims=True)
        region, _ = asignar(Z, self.centroides_)
        conteo = np.bincount(y_cod, minlength=C).astype(float)
        s = (n / (C * np.maximum(conteo, 1))) ** self.gamma
        s = s[y_cod]
        s = s / s.mean()
        Y = np.eye(C)[y_cod]
        n_r = np.bincount(region, minlength=R).astype(float)
        peso_r = n_r / n
        x0 = np.zeros((R + 1) * C * d)
        reg_mask = np.ones(d)
        reg_mask[-1] = 0.0          # el termino independiente global no se penaliza

        def objetivo(theta):
            T = theta.reshape(R + 1, C, d)
            T0, Tr = T[0], T[1:]
            logits = np.einsum("nd,ncd->nc", F, Tr[region])
            lp = log_softmax(logits, axis=1)
            perdida = -(s * (Y * lp).sum(1)).sum() / n
            G = (np.exp(lp) - Y) * s[:, None] / n                  # dL/dlogits
            grad_r = np.zeros_like(Tr)
            np.add.at(grad_r, region, G[:, :, None] * F[:, None, :])
            dif = Tr - T0[None]
            perdida += 0.5 * self.lam0 * ((T0 ** 2) * reg_mask).sum() / n
            perdida += 0.5 * self.lam * (peso_r[:, None, None] * dif ** 2).sum() / n
            grad_r += self.lam * peso_r[:, None, None] * dif / n
            grad0 = self.lam0 * T0 * reg_mask / n - self.lam * (peso_r[:, None, None] * dif).sum(0) / n
            return perdida, np.concatenate([grad0.ravel(), grad_r.ravel()])

        res = minimize(objetivo, x0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        T = res.x.reshape(R + 1, C, d)
        self.theta_global_, self.theta_regiones_ = T[0], T[1:]
        self.convergio_, self.n_iter_, self.objetivo_final_ = bool(res.success), int(res.nit), float(res.fun)
        self.n_estimadores_ = P_oof.shape[0]
        return self

    # ------------------------------------------------------------------ prediccion
    def regiones(self, X):
        check_is_fitted(self, "centroides_")
        return asignar(self._espacio(X), self.centroides_)[0]

    def combinar(self, P, X):
        """Probabilidades combinadas a partir de las de los modelos base P (M, n, C)."""
        check_is_fitted(self, "theta_regiones_")
        F = caracteristicas(np.asarray(P, dtype=np.float64))
        r = self.regiones(X)
        logits = np.einsum("nd,ncd->nc", F, self.theta_regiones_[r])
        return np.exp(log_softmax(logits, axis=1))

    def predict_proba(self, X):
        check_is_fitted(self, "estimadores_")
        X_base = X if hasattr(X, "iloc") else check_array(X, dtype=np.float64)
        P = np.stack([self._proba_base(est, X_base) for _, est in self.estimadores_])
        return self.combinar(P, X_base)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
