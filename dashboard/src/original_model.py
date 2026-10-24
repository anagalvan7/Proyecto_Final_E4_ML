"""Ensamble por competencia local y por clase (ECLC).

Modelo original del proyecto. Combina M clasificadores base ya existentes con una
regla de ponderacion propia: para cada observacion nueva x y cada clase c, el peso del
modelo m depende de su perdida logaritmica binaria (clase c contra el resto) en los k
vecinos mas cercanos de x dentro de un conjunto de referencia con predicciones fuera de
pliegue (out-of-fold, OOF).

Notacion (capitulo 5 del libro):
    P_m(x) in R^C                probabilidades del modelo base m
    l_mc(j) = -[y_j=c] log p_mc(j) - [y_j!=c] log(1 - p_mc(j))   perdida binaria OOF
    a_j(x) = s_{y_j} exp(-(d_j / d_k)^2)                          peso del vecino j
    Lloc_mc(x) = sum_j a_j l_mc(j) / A(x),   A(x) = sum_j a_j
    Lglob_mc   = sum_i s_{y_i} l_mc(i) / sum_i s_{y_i}
    L_mc(x)    = (A(x) Lloc_mc(x) + tau Lglob_mc) / (A(x) + tau)      (contraccion)
    w_mc(x)    = exp(-beta L_mc(x)) / sum_m' exp(-beta L_m'c(x))     (regla de Gibbs)
    q_c(x)     = sum_m w_mc(x) p_mc(x);   p(c|x) = q_c(x) / sum_c' q_c'(x)

w_c(x) es la solucion unica de  min_{w en el simplex}  sum_m w_m L_mc(x) + (1/beta) sum_m w_m log w_m,
lo que da la cota  L_c(q) <= sum_m w_mc L_mc <= min_m L_mc + log(M)/beta  (Proposiciones 1 y 2).

Solo usa numpy para la regla de combinacion y para la busqueda de vecinos; los modelos
base pueden venir de cualquier libreria compatible con scikit-learn (respuesta del
docente a la consulta sobre la propuesta).
"""
from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

EPS = 1e-6


def _estandarizar_ajuste(Z):
    """Media y desviacion de cada columna (las constantes quedan con escala 1)."""
    media = Z.mean(axis=0)
    escala = Z.std(axis=0)
    escala[escala < 1e-12] = 1.0
    return media, escala


def vecinos_mas_cercanos(Z_ref, Z_q, k, tam_bloque=2048):
    """Busqueda exacta de los k vecinos (distancia euclidiana) por bloques.

    Usa ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b para calcular un bloque de distancias con
    un producto de matrices y np.argpartition para extraer los k menores sin ordenar todo
    el bloque. Costo O(n_q * n_ref * p) en tiempo y O(tam_bloque * n_ref) en memoria.

    Devuelve (indices, distancias), ambos de forma (n_q, k), ordenados de menor a mayor.
    """
    n_ref = Z_ref.shape[0]
    k = min(k, n_ref)
    norma_ref = np.einsum("ij,ij->i", Z_ref, Z_ref)
    idx = np.empty((Z_q.shape[0], k), dtype=np.int64)
    dist = np.empty((Z_q.shape[0], k), dtype=np.float64)
    for ini in range(0, Z_q.shape[0], tam_bloque):
        B = Z_q[ini:ini + tam_bloque]
        d2 = norma_ref[None, :] + np.einsum("ij,ij->i", B, B)[:, None] - 2.0 * (B @ Z_ref.T)
        np.maximum(d2, 0.0, out=d2)
        if k < n_ref:
            parte = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            parte = np.tile(np.arange(n_ref), (B.shape[0], 1))
        d_parte = np.take_along_axis(d2, parte, axis=1)
        orden = np.argsort(d_parte, axis=1, kind="stable")
        idx[ini:ini + len(B)] = np.take_along_axis(parte, orden, axis=1)
        dist[ini:ini + len(B)] = np.sqrt(np.take_along_axis(d_parte, orden, axis=1))
    return idx, dist


def perdidas_binarias(P, y_cod, n_clases):
    """l[j, m, c]: perdida logaritmica binaria (c contra el resto) del modelo m en el punto j.

    P tiene forma (M, n, C); y_cod son las etiquetas codificadas 0..C-1.
    """
    P = np.clip(P, EPS, 1.0 - EPS)
    Y = np.eye(n_clases, dtype=bool)[y_cod]                     # (n, C)
    l = -np.where(Y[None, :, :], np.log(P), np.log1p(-P))       # (M, n, C)
    return np.ascontiguousarray(l.transpose(1, 0, 2))           # (n, M, C)


def perdidas_multiclase(P, y_cod):
    """l[j, m]: perdida logaritmica multiclase del modelo m en el punto j (variante sin clase)."""
    P = np.clip(P, EPS, 1.0)
    l = -np.log(P[:, np.arange(len(y_cod)), y_cod])             # (M, n)
    return np.ascontiguousarray(l.T)                            # (n, M)


def pesos_gibbs(L, beta, eje):
    """Solucion cerrada w = softmax(-beta L) a lo largo del eje de los modelos."""
    S = -beta * L
    S = S - S.max(axis=eje, keepdims=True)
    W = np.exp(S)
    return W / W.sum(axis=eje, keepdims=True)


class EnsambleCompetenciaLocal(ClassifierMixin, BaseEstimator):
    """Ensamble por competencia local y por clase (ECLC).

    El mixin va antes de BaseEstimator: desde scikit-learn 1.6 las etiquetas del
    estimador (is_classifier, validacion cruzada estratificada por defecto) se resuelven
    por el orden de herencia, y con el orden inverso la clase no se reconoce como
    clasificador.

    Parameters
    ----------
    estimadores : list of (str, estimator)
        Clasificadores base compatibles con scikit-learn que implementan predict_proba.
    k : int
        Numero de vecinos del conjunto de referencia que definen la competencia local.
    beta : float >= 0
        Inverso de la temperatura de la regla de Gibbs. beta = 0 da el promedio simple;
        beta -> infinito selecciona el modelo con menor perdida local.
    tau : float >= 0 o np.inf
        Contraccion hacia la perdida global de cada modelo y clase. tau = 0 usa solo la
        vecindad; tau = inf da pesos globales por clase (sin componente local).
    por_clase : bool
        True: un vector de pesos por clase con la perdida binaria c-contra-resto.
        False: un solo vector de pesos con la perdida logaritmica multiclase (ablacion).
    ponderar_distancia : bool
        True: nucleo gaussiano adaptativo exp(-(d_j/d_k)^2). False: vecinos equiponderados.
    balancear_clases : bool
        True: cada vecino pesa n / (C n_c) para que las clases raras cuenten en el F1 macro.
    n_pliegues : int
        Pliegues para las predicciones OOF que forman el conjunto de referencia.
    transformador_vecinos : transformer o None
        Transformacion (se ajusta en fit) del espacio donde se buscan vecinos. Si es None,
        se estandarizan todas las columnas con media y desviacion del entrenamiento.
    random_state : int
        Semilla de los pliegues OOF y de los estimadores base que la admiten.
    tam_bloque : int
        Filas por bloque en la busqueda de vecinos (controla la memoria).
    """

    def __init__(self, estimadores=None, k=50, beta=5.0, tau=10.0, por_clase=True,
                 ponderar_distancia=True, balancear_clases=True, n_pliegues=3,
                 transformador_vecinos=None, random_state=42, tam_bloque=2048):
        self.estimadores = estimadores
        self.k = k
        self.beta = beta
        self.tau = tau
        self.por_clase = por_clase
        self.ponderar_distancia = ponderar_distancia
        self.balancear_clases = balancear_clases
        self.n_pliegues = n_pliegues
        self.transformador_vecinos = transformador_vecinos
        self.random_state = random_state
        self.tam_bloque = tam_bloque

    # ------------------------------------------------------------------ validacion
    def _validar_hiperparametros(self):
        if not self.estimadores:
            raise ValueError("Se requiere al menos un estimador base en 'estimadores'.")
        if not isinstance(self.k, numbers.Integral) or self.k < 1:
            raise ValueError(f"k debe ser un entero >= 1; se recibio {self.k!r}.")
        if not isinstance(self.beta, numbers.Real) or not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError(f"beta debe ser un real finito >= 0; se recibio {self.beta!r}.")
        if not isinstance(self.tau, numbers.Real) or np.isnan(self.tau) or self.tau < 0:
            raise ValueError(f"tau debe ser >= 0 (se admite np.inf); se recibio {self.tau!r}.")
        if not isinstance(self.n_pliegues, numbers.Integral) or self.n_pliegues < 2:
            raise ValueError(f"n_pliegues debe ser un entero >= 2; se recibio {self.n_pliegues!r}.")

    def _matriz(self, X):
        """Arreglo numerico para validar; los estimadores base reciben X sin modificar."""
        return check_array(X, dtype=np.float64, ensure_all_finite=True)

    def _espacio(self, X, ajustar=False):
        """Coordenadas del espacio de vecinos (float32 para abaratar la busqueda)."""
        if self.transformador_vecinos is not None:
            if ajustar:
                self.transformador_vecinos_ = clone(self.transformador_vecinos).fit(X)
            Z = np.asarray(self.transformador_vecinos_.transform(X), dtype=np.float64)
        else:
            Z = self._matriz(X)
            if ajustar:
                self.media_, self.escala_ = _estandarizar_ajuste(Z)
            Z = (Z - self.media_) / self.escala_
        return np.ascontiguousarray(Z, dtype=np.float32)

    def _proba_base(self, est, X):
        """predict_proba de un estimador base con las columnas alineadas a classes_."""
        P = est.predict_proba(X)
        clases_est = getattr(est, "classes_", None)
        if clases_est is None or np.array_equal(clases_est, self.clases_codificadas_):
            return P
        salida = np.zeros((P.shape[0], len(self.classes_)))
        salida[:, np.asarray(clases_est, dtype=int)] = P
        return salida

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X, y):
        """Ajusta los modelos base, construye las predicciones OOF y la regla de pesos."""
        self._validar_hiperparametros()
        Xv, y = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        X_base = X if hasattr(X, "iloc") else Xv
        self.classes_, y_cod = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Se requieren al menos dos clases en y.")
        self.clases_codificadas_ = np.arange(len(self.classes_))
        n, C, M = len(y_cod), len(self.classes_), len(self.estimadores)

        # 1) Predicciones fuera de pliegue de cada modelo base sobre el entrenamiento.
        P_oof = np.zeros((M, n, C))
        cv = StratifiedKFold(self.n_pliegues, shuffle=True, random_state=self.random_state)
        for tr, va in cv.split(Xv, y_cod):
            X_tr = X_base.iloc[tr] if hasattr(X_base, "iloc") else X_base[tr]
            X_va = X_base.iloc[va] if hasattr(X_base, "iloc") else X_base[va]
            for m, (_, est) in enumerate(self.estimadores):
                e = clone(est).fit(X_tr, y_cod[tr])
                P_oof[m, va] = self._proba_base(e, X_va)
        # 2) Modelos base finales con todo el entrenamiento.
        self.estimadores_ = [(nombre, clone(est).fit(X_base, y_cod)) for nombre, est in self.estimadores]
        # 3) Regla de combinacion.
        return self.ajustar_desde_oof(X_base, y, P_oof, ajustar_espacio=True)

    def ajustar_desde_oof(self, X, y, P_oof, ajustar_espacio=True):
        """Ajusta solo la capa de combinacion a partir de probabilidades OOF ya calculadas.

        Permite optimizar k, beta y tau sin reentrenar los modelos base (los modelos base
        no dependen de esos hiperparametros). P_oof tiene forma (M, n, C).
        """
        self._validar_hiperparametros()
        y = np.asarray(y)
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(y)
            self.clases_codificadas_ = np.arange(len(self.classes_))
        y_cod = np.searchsorted(self.classes_, y)
        C = len(self.classes_)
        P_oof = np.asarray(P_oof, dtype=np.float64)
        if P_oof.ndim != 3 or P_oof.shape[1] != len(y) or P_oof.shape[2] != C:
            raise ValueError(f"P_oof debe tener forma (M, {len(y)}, {C}); tiene {P_oof.shape}.")
        if ajustar_espacio:
            self.Z_ref_ = self._espacio(X, ajustar=True)
        self.y_ref_ = y_cod
        conteo = np.bincount(y_cod, minlength=C).astype(float)
        if self.balancear_clases:
            self.peso_clase_ = len(y_cod) / (C * np.maximum(conteo, 1.0))
        else:
            self.peso_clase_ = np.ones(C)
        s = self.peso_clase_[y_cod]
        if self.por_clase:
            self.perdidas_ref_ = perdidas_binarias(P_oof, y_cod, C).astype(np.float32)   # (n, M, C)
            self.perdida_global_ = np.einsum("j,jmc->mc", s, self.perdidas_ref_) / s.sum()
        else:
            self.perdidas_ref_ = perdidas_multiclase(P_oof, y_cod).astype(np.float32)    # (n, M)
            self.perdida_global_ = (s @ self.perdidas_ref_) / s.sum()
        self.n_estimadores_ = P_oof.shape[0]
        if self.transformador_vecinos is None:
            self.n_features_in_ = self.Z_ref_.shape[1]
        return self

    # ------------------------------------------------------------------ prediccion
    def perdidas_locales(self, Z, vecinos=None):
        """Perdida contraida L_mc(x) de cada modelo (y clase) en la vecindad de cada fila de Z.

        vecinos : (idx, dist) opcional, ya calculados con vecinos_mas_cercanos sobre Z_ref_ con al
        menos k columnas (ordenadas de menor a mayor distancia). Permite reutilizar una sola busqueda
        para varios valores de (k, beta, tau): el resultado es identico al de buscar de nuevo.
        """
        n_q = Z.shape[0] if Z is not None else vecinos[0].shape[0]
        if np.isinf(self.tau):
            forma = (n_q,) + self.perdida_global_.shape
            return np.broadcast_to(self.perdida_global_, forma).astype(np.float64)
        if vecinos is not None and vecinos[0].shape[1] < min(self.k, len(self.y_ref_)):
            raise ValueError(f"Los vecinos precalculados tienen {vecinos[0].shape[1]} columnas; se requieren {self.k}.")
        # Por bloques de consultas: el tensor de perdidas de los vecinos tiene forma
        # (n_q, k, M, C) y para k grande no cabe completo en memoria.
        bloques = []
        for i in range(0, n_q, self.tam_bloque):
            vb = None if vecinos is None else (vecinos[0][i:i + self.tam_bloque, :self.k],
                                               vecinos[1][i:i + self.tam_bloque, :self.k])
            bloques.append(self._perdidas_bloque(None if Z is None else Z[i:i + self.tam_bloque], vb))
        return np.concatenate(bloques)

    def _perdidas_bloque(self, Z, vecinos=None):
        if vecinos is None:
            idx, dist = vecinos_mas_cercanos(self.Z_ref_, Z, self.k, self.tam_bloque)
        else:
            idx, dist = vecinos
        if self.ponderar_distancia:
            escala = np.maximum(dist[:, -1:], 1e-12)
            a = np.exp(-(dist / escala) ** 2)
        else:
            a = np.ones_like(dist)
        a = a * self.peso_clase_[self.y_ref_[idx]]                     # (n_q, k)
        A = a.sum(axis=1)
        l_vec = self.perdidas_ref_[idx]                                 # (n_q, k, M[, C])
        if self.por_clase:
            L_loc = np.einsum("qj,qjmc->qmc", a, l_vec) / A[:, None, None]
            return (A[:, None, None] * L_loc + self.tau * self.perdida_global_) / (A[:, None, None] + self.tau)
        L_loc = np.einsum("qj,qjm->qm", a, l_vec) / A[:, None]
        return (A[:, None] * L_loc + self.tau * self.perdida_global_) / (A[:, None] + self.tau)

    def pesos(self, X=None, Z=None, vecinos=None):
        """Pesos w_mc(x) de forma (n, M, C) (o (n, M) si por_clase=False)."""
        check_is_fitted(self, "perdidas_ref_")
        if Z is None and vecinos is None:
            Z = self._espacio(X)
        L = self.perdidas_locales(Z, vecinos)
        return pesos_gibbs(L, self.beta, eje=1)

    def combinar(self, P, X=None, Z=None, vecinos=None):
        """Combina probabilidades base P (M, n, C) con la regla ECLC."""
        W = self.pesos(X=X, Z=Z, vecinos=vecinos)
        if not self.por_clase:
            W = np.repeat(W[:, :, None], P.shape[2], axis=2)
        q = np.einsum("qmc,mqc->qc", W, P)
        return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-300)

    def probas_base(self, X):
        """Probabilidades de cada modelo base ajustado, forma (M, n, C)."""
        check_is_fitted(self, "estimadores_")
        return np.stack([self._proba_base(est, X) for _, est in self.estimadores_])

    def predict_proba(self, X):
        check_is_fitted(self, "perdidas_ref_")
        Xv = self._matriz(X)
        if self.transformador_vecinos is None and Xv.shape[1] != self.Z_ref_.shape[1]:
            raise ValueError(f"X tiene {Xv.shape[1]} columnas; el modelo se ajusto con {self.Z_ref_.shape[1]}.")
        X_base = X if hasattr(X, "iloc") else Xv
        P = self.probas_base(X_base)
        return self.combinar(P, X=X_base)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
