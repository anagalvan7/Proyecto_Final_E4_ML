"""Utilidades del benchmark combinatorio de modelos de clasificación.

Covertype es un problema puramente de clasificación multiclase (Cover_Type, 7 clases
nominales), sin variable objetivo continua. Por eso el benchmark se restringe a la
clasificacion (7 modelos x 4 balanceo x 4 optimizacion = 112 combinaciones); la
regresion no tiene un objetivo natural en este conjunto de datos (capitulo 5).

Para mantener el costo computacional viable en una sola maquina y en el tiempo
disponible, el benchmark completo (112 combinaciones x validacion anidada) se corre
sobre una submuestra estratificada, no sobre las 464,809 filas de entrenamiento
completas. El capitulo 3 ya establecio y justifico este mismo criterio (la curva de
aprendizaje muestra que el F1 de validacion se estabiliza desde ~30,000 filas); aqui se
reutiliza la misma lógica para 112 combinaciones en lugar de 12.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
from _utils import (  # noqa: E402
    RANDOM_STATE, load_covertype, split_covertype, stratified_sample, engineer_features,
    NUM_COLS, BIN_COLS, ENG_NUM_COLS, CLASS_SHORT, CLASS_PALETTE, CLASS_NAMES,
)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Tamanos de submuestra, documentados y justificados por costo computacional (ver cap. 6)
# --------------------------------------------------------------------------------------
N_BENCHMARK = 40_000        # grid de 112 combinaciones, validacion anidada
N_SVM = 9_000               # SVM con nucleo RBF escala mal (O(n^2)-O(n^3)); submuestra aparte
N_FINALISTS = 120_000       # reentrenamiento de los finalistas (top modelos) para el informe final

OUTER_FOLDS = 3
INNER_FOLDS = 3

CLASES = list(range(1, 8))


def get_data(n=N_BENCHMARK, seed=RANDOM_STATE):
    """Carga Covertype y devuelve una submuestra estratificada de entrenamiento + prueba completa."""
    df, X, y = load_covertype()
    X_train_full, X_test, y_train_full, y_test = split_covertype(X, y)
    X_train, y_train = stratified_sample(X_train_full, y_train_full, n, random_state=seed)
    return X_train, y_train, X_test, y_test, X_train_full, y_train_full


def build_preprocessor_steps():
    """Mismo preprocesamiento de la Parte I (capitulo 2.9): ingenieria + escalado de continuas.

    Devuelve los PASOS (no un Pipeline anidado), porque imblearn.Pipeline no admite un
    Pipeline de scikit-learn como paso intermedio: hay que aplanarlos.
    """
    scale = ColumnTransformer(
        [("num", StandardScaler(), ENG_NUM_COLS), ("bin", "passthrough", BIN_COLS)]
    )
    return [("features", FunctionTransformer(engineer_features)), ("scale", scale)]


# --------------------------------------------------------------------------------------
# Etapa 1: los 7 modelos base y sus espacios de busqueda (realistas, no arbitrarios)
# --------------------------------------------------------------------------------------
# Los rangos se fijaron a partir de la escala del problema (57 predictoras, ~40,000-460,000
# filas, 7 clases) y de los valores que ya funcionaron en el capitulo 3 (C=1 para la
# logistica), no de valores arbitrarios.

MODELOS = {
    "KNN": {
        "build": lambda **kw: KNeighborsClassifier(n_jobs=1, **kw),
        "espacio": {"n_neighbors": [5, 11, 21, 41], "weights": ["uniform", "distance"]},
        "tipo": {"n_neighbors": "int", "weights": "cat"},
        "soporta_class_weight": False,
        "soporta_sample_weight": False,
    },
    "NB": {
        "build": lambda **kw: GaussianNB(**kw),
        "espacio": {"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
        "tipo": {"var_smoothing": "logfloat"},
        "soporta_class_weight": False,
        "soporta_sample_weight": True,
    },
    "LR": {
        "build": lambda **kw: LogisticRegression(
            solver="saga", max_iter=150, tol=1e-2, random_state=RANDOM_STATE, **kw
        ),
        "espacio": {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
        "tipo": {"C": "logfloat", "penalty": "cat"},
        "soporta_class_weight": True,
        "soporta_sample_weight": True,
    },
    "DT": {
        "build": lambda **kw: DecisionTreeClassifier(random_state=RANDOM_STATE, **kw),
        "espacio": {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 10, 50]},
        "tipo": {"max_depth": "int_none", "min_samples_leaf": "int"},
        "soporta_class_weight": True,
        "soporta_sample_weight": True,
    },
    "RF": {
        "build": lambda **kw: RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, **kw
        ),
        "espacio": {"n_estimators": [100, 200], "max_depth": [10, 20]},
        "tipo": {"n_estimators": "int", "max_depth": "int_none"},
        "soporta_class_weight": True,
        "soporta_sample_weight": True,
    },
    "XGB": {
        "build": lambda **kw: XGBClassifier(
            objective="multi:softprob", num_class=7, eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=1, **kw
        ),
        "espacio": {"n_estimators": [100, 150], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        "tipo": {"n_estimators": "int", "max_depth": "int", "learning_rate": "logfloat"},
        "soporta_class_weight": False,
        "soporta_sample_weight": True,
    },
    "SVM": {
        "build": lambda **kw: SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE, **kw),
        "espacio": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
        "tipo": {"C": "logfloat", "gamma": "cat"},
        "soporta_class_weight": True,
        "soporta_sample_weight": True,
    },
}
NOMBRES_MODELOS = {
    "KNN": "K vecinos más cercanos", "NB": "Naive Bayes gaussiano", "LR": "Regresión logística (L1/L2)",
    "DT": "Árbol de decisión", "RF": "Random Forest", "XGB": "XGBoost", "SVM": "SVM (núcleo RBF)",
}

BALANCEO = ["ninguno", "smote", "adasyn", "balanced"]
NOMBRES_BALANCEO = {
    "ninguno": "Sin balanceo (línea base)", "smote": "SMOTE", "adasyn": "ADASYN",
    "balanced": "class_weight / sample_weight balanceado",
}

OPTIMIZACION = ["grid", "random", "bayes", "genetico"]
NOMBRES_OPT = {
    "grid": "Grid Search", "random": "Random Search",
    "bayes": "Bayesiana (Optuna, TPE)", "genetico": "Genético (DEAP)",
}


def build_estimator_pipeline(clave_modelo, balanceo, k_vecinos=3):
    """Arma el pipeline (preprocesamiento [+ balanceo] + modelo) para una combinación dada.

    k_vecinos se ajusta dinámicamente en `robust_fit` según la clase menos frecuente del
    pliegue de entrenamiento: SMOTE/ADASYN exigen al menos k_vecinos+1 muestras de esa clase.
    """
    base = MODELOS[clave_modelo]["build"]()
    steps = list(build_preprocessor_steps())
    if balanceo == "smote":
        steps.append(("bal", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_vecinos)))
    elif balanceo == "adasyn":
        steps.append(("bal", ADASYN(random_state=RANDOM_STATE, n_neighbors=k_vecinos)))
    steps.append(("clf", base))
    Pipe = ImbPipeline if balanceo in ("smote", "adasyn") else SkPipeline
    return Pipe(steps)


def robust_fit(clave_modelo, balanceo, params, X, y, fit_params=None):
    """Ajusta la combinación (modelo, balanceo) con reintentos ante clases demasiado
    pequeñas para SMOTE/ADASYN en un pliegue concreto.

    Si la clase minoritaria del pliegue tiene menos de 2 observaciones, SMOTE/ADASYN no
    pueden generar vecinos: se omite el remuestreo para ESE pliegue y se deja constancia
    (campo 'remuestreo_omitido'), en vez de que el proceso completo falle.
    """
    fit_params = fit_params or {}
    params = _resolve_params(clave_modelo, params)
    conteo_min = y.value_counts().min()
    k_efectivo = max(1, min(3, conteo_min - 1))
    remuestreo_omitido = False
    bal_efectivo = balanceo
    if balanceo in ("smote", "adasyn") and conteo_min < 2:
        bal_efectivo = "ninguno"
        remuestreo_omitido = True
    pipe = build_estimator_pipeline(clave_modelo, bal_efectivo, k_vecinos=k_efectivo)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    if bal_efectivo == "balanced":
        info = MODELOS[clave_modelo]
        if info["soporta_class_weight"]:
            pipe.set_params(clf__class_weight="balanced")
    try:
        pipe.fit(X, y, **fit_params)
    except TypeError:
        pipe.fit(X, y)
    except ValueError as e:
        if "n_neighbors" in str(e) and balanceo in ("smote", "adasyn"):
            pipe = build_estimator_pipeline(clave_modelo, "ninguno")
            pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
            try:
                pipe.fit(X, y, **fit_params)
            except TypeError:
                pipe.fit(X, y)
            remuestreo_omitido = True
        else:
            raise
    return pipe, remuestreo_omitido


def aplica_balanceo(clave_modelo, balanceo, pipe, X, y):
    """Para 'balanced': fija class_weight si el modelo lo soporta, o calcula sample_weight.

    Devuelve (pipe_configurado, fit_kwargs, aplicable). aplicable=False si el modelo no
    soporta ninguna forma de ponderar clases (ej. KNN): ese caso se documenta y se omite.
    """
    if balanceo != "balanced":
        return pipe, {}, True
    info = MODELOS[clave_modelo]
    if info["soporta_class_weight"]:
        pipe.set_params(clf__class_weight="balanced")
        return pipe, {}, True
    if info["soporta_sample_weight"]:
        return pipe, {}, True  # el peso se calcula por fold en la función de evaluación
    return pipe, {}, False


def sample_weight_for(y):
    return compute_sample_weight("balanced", y)


# --------------------------------------------------------------------------------------
# Puntaje de una configuración de hiperparámetros con validación cruzada interna
# --------------------------------------------------------------------------------------

def _fit_params_for(clave_modelo, balanceo, y_fit):
    """Prefijo del paso del pipeline + sample_weight si aplica (balanced y no hay class_weight)."""
    fit_params = {}
    if balanceo == "balanced" and not MODELOS[clave_modelo]["soporta_class_weight"]:
        sw = sample_weight_for(y_fit)
        fit_params["clf__sample_weight"] = sw
    return fit_params


def _resolve_params(clave_modelo, params):
    """Ajustes dependientes de otros hiperparámetros (ej. LR: lbfgs es ~10x más rápido que
    saga para L2, pero lbfgs no admite L1; se elige el solver según la penalización)."""
    params = dict(params)
    if clave_modelo == "LR":
        params["solver"] = "saga" if params.get("penalty") == "l1" else "lbfgs"
    return params


def cv_score(clave_modelo, balanceo, params, X, y, cv):
    """F1 macro promedio en validación cruzada interna para una configuración de hiperparámetros."""
    if balanceo == "balanced" and not MODELOS[clave_modelo]["soporta_class_weight"] \
            and not MODELOS[clave_modelo]["soporta_sample_weight"]:
        return np.nan
    from sklearn.metrics import f1_score
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        # XGBoost exige etiquetas 0..k-1; Cover_Type viene en 1..7, se recorre solo para XGB.
        ytr_fit = (ytr - 1) if clave_modelo == "XGB" else ytr
        fit_params = _fit_params_for(clave_modelo, balanceo, ytr_fit)
        pipe, _ = robust_fit(clave_modelo, balanceo, params, Xtr, ytr_fit, fit_params)
        pred = pipe.predict(Xva)
        if clave_modelo == "XGB":
            pred = pred + 1
        scores.append(f1_score(yva, pred, average="macro", zero_division=0))
    return float(np.mean(scores))


# --------------------------------------------------------------------------------------
# Los cuatro métodos de optimización, cada uno con un presupuesto reducido y documentado
# --------------------------------------------------------------------------------------

def grid_search(clave_modelo, balanceo, X, y, cv):
    espacio = MODELOS[clave_modelo]["espacio"]
    from itertools import product
    claves = list(espacio.keys())
    combos = list(product(*[espacio[k] for k in claves]))
    historia = []
    for combo in combos:
        params = dict(zip(claves, combo))
        s = cv_score(clave_modelo, balanceo, params, X, y, cv)
        historia.append({"params": params, "score": s})
    historia = [h for h in historia if not np.isnan(h["score"])]
    if not historia:
        return None, [], 0
    mejor = max(historia, key=lambda h: h["score"])
    return mejor, historia, len(combos)


def random_search(clave_modelo, balanceo, X, y, cv, n_iter=10, seed=RANDOM_STATE):
    espacio = MODELOS[clave_modelo]["espacio"]
    rng = np.random.default_rng(seed)
    claves = list(espacio.keys())
    historia = []
    for _ in range(n_iter):
        params = {k: espacio[k][rng.integers(0, len(espacio[k]))] for k in claves}
        s = cv_score(clave_modelo, balanceo, params, X, y, cv)
        historia.append({"params": params, "score": s})
    historia = [h for h in historia if not np.isnan(h["score"])]
    if not historia:
        return None, [], 0
    mejor = max(historia, key=lambda h: h["score"])
    return mejor, historia, n_iter


def bayes_search(clave_modelo, balanceo, X, y, cv, n_trials=12, seed=RANDOM_STATE):
    """Optimización bayesiana con Optuna (TPE = Tree-structured Parzen Estimator)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    espacio = MODELOS[clave_modelo]["espacio"]
    tipos = MODELOS[clave_modelo]["tipo"]
    historia = []

    def objective(trial):
        params = {}
        for k, valores in espacio.items():
            t = tipos[k]
            if t == "cat":
                params[k] = trial.suggest_categorical(k, valores)
            elif t == "int":
                params[k] = trial.suggest_categorical(k, valores)
            elif t == "int_none":
                opciones = [str(v) for v in valores]
                elegido = trial.suggest_categorical(k, opciones)
                params[k] = None if elegido == "None" else int(elegido)
            elif t == "logfloat":
                lo, hi = min(valores), max(valores)
                params[k] = trial.suggest_float(k, lo, hi, log=(lo > 0))
        s = cv_score(clave_modelo, balanceo, params, X, y, cv)
        historia.append({"params": params, "score": s})
        return s if not np.isnan(s) else -1.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    historia_validos = [h for h in historia if not np.isnan(h["score"])]
    if not historia_validos:
        return None, [], 0
    mejor = max(historia_validos, key=lambda h: h["score"])
    return mejor, historia_validos, n_trials


def genetic_search(clave_modelo, balanceo, X, y, cv, pop_size=8, n_gen=4, seed=RANDOM_STATE):
    """Algoritmo genético con DEAP: selección por torneo, cruce uniforme, mutación por reemplazo."""
    from deap import base, creator, tools
    import random as pyrandom

    espacio = MODELOS[clave_modelo]["espacio"]
    claves = list(espacio.keys())
    rng = pyrandom.Random(seed)
    historia = []
    diversidad_por_gen = []

    def individuo_aleatorio():
        return [rng.randrange(len(espacio[k])) for k in claves]

    def params_de(ind):
        return {k: espacio[k][idx] for k, idx in zip(claves, ind)}

    def evaluar(ind):
        params = params_de(ind)
        s = cv_score(clave_modelo, balanceo, params, X, y, cv)
        historia.append({"params": params, "score": s})
        return (s if not np.isnan(s) else -1.0,)

    if "FitnessMax3" not in dir(creator):
        creator.create("FitnessMax3", base.Fitness, weights=(1.0,))
        creator.create("Individual3", list, fitness=creator.FitnessMax3)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual3, individuo_aleatorio)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def mutar(ind):
        i = rng.randrange(len(ind))
        ind[i] = rng.randrange(len(espacio[claves[i]]))
        return (ind,)
    toolbox.register("mutate", mutar)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    n_elite = 1
    for gen in range(n_gen):
        valores_unicos = len({tuple(ind) for ind in pop})
        diversidad_por_gen.append({"generacion": gen, "individuos_unicos": valores_unicos, "tam_poblacion": len(pop)})
        elite = tools.selBest(pop, n_elite)
        descendencia = toolbox.select(pop, len(pop) - n_elite)
        descendencia = [toolbox.clone(ind) for ind in descendencia]
        for c1, c2 in zip(descendencia[::2], descendencia[1::2]):
            if rng.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for ind in descendencia:
            if rng.random() < 0.3:
                toolbox.mutate(ind)
                del ind.fitness.values
        for ind in descendencia:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = elite + descendencia

    historia_validos = [h for h in historia if not np.isnan(h["score"])]
    if not historia_validos:
        return None, [], 0, diversidad_por_gen
    mejor = max(historia_validos, key=lambda h: h["score"])
    return mejor, historia_validos, len(historia_validos), diversidad_por_gen


OPTIMIZADORES = {
    "grid": lambda cm, b, X, y, cv: (lambda r: (r[0], r[1], r[2]))(grid_search(cm, b, X, y, cv)),
    "random": lambda cm, b, X, y, cv: random_search(cm, b, X, y, cv, n_iter=6),
    "bayes": lambda cm, b, X, y, cv: bayes_search(cm, b, X, y, cv, n_trials=6),
    "genetico": lambda cm, b, X, y, cv: (lambda r: (r[0], r[1], r[2]))(
        genetic_search(cm, b, X, y, cv, pop_size=4, n_gen=2)
    ),
}
