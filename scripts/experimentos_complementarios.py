"""Experimentos complementarios del capítulo de optimización y del de complejidad.

Genera, en ``resultados/``, los insumos de cuatro análisis que el
benchmark combinatorio no registró:

1. ``anytime``: curvas de desempeño en cualquier momento (*anytime performance*) de
   Grid, Random, Bayesiano (TPE) y Genético con un presupuesto idéntico de
   evaluaciones, repetidas con cinco semillas (estabilidad de la solución).
2. ``computacional``: comparación entre la implementación estándar y la optimizada de
   cada algoritmo (tiempo de entrenamiento, tiempo de inferencia, memoria pico y F1).
3. ``escalamiento``: tiempo de entrenamiento en función de *n* para estimar el
   exponente empírico de la curva de escalamiento.
4. ``shap``: valores SHAP del finalista sobre una muestra de prueba, para los
   gráficos de resumen (*beeswarm*) y de cascada (*waterfall*).

Uso::

    python experimentos_complementarios.py anytime computacional escalamiento shap areas figuras_es lime

Todas las semillas derivan de ``RANDOM_STATE = 42``; los tiempos son de reloj de pared
(``time.perf_counter``) y la memoria pico se mide con ``tracemalloc``.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline as SkPipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "resultados"
RESULTS.mkdir(exist_ok=True)
SEED = u3.RANDOM_STATE
PRESUPUESTO = 30          # evaluaciones por corrida (idéntico para los cuatro métodos)
SEMILLAS = [0, 1, 2, 3, 4]


# Configuracion de XGBoost para SHAP, LIME y areas no vistas. None: la del finalista guardado en
# resumen_finalista.json; scripts/e3_xgb_espacio_comun.py la fija a la mejor de XGBoost en el espacio comun.
HP_XGB = None


def _hp_xgb():
    if HP_XGB is not None:
        return dict(HP_XGB)
    with open(RESULTS / "resumen_finalista.json", encoding="utf-8") as f:
        return json.load(f)["hiperparametros"]


def _entorno():
    import os
    return {"python": platform.python_version(), "sistema": platform.platform(),
            "nucleos_logicos": os.cpu_count()}


# ---------------------------------------------------------------------------------------
# 1. Anytime performance y estabilidad ante semillas
# ---------------------------------------------------------------------------------------
# Espacio mixto de cinco dimensiones para un árbol de decisión. Grid Search solo puede
# cubrir dos dimensiones con 30 puntos (6 x 5), mientras que Random, TPE y el genético
# exploran las cinco: es precisamente el escenario en el que la búsqueda dirigida debería
# rendir más por evaluación.
ESPACIO_DT = {
    "max_depth": (3, 40),                 # entero
    "min_samples_leaf": (1, 200),         # entero, escala logarítmica
    "min_samples_split": (2, 100),        # entero, escala logarítmica
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt", 0.5],
}
GRID_DT = {"max_depth": [5, 10, 15, 20, 30, 40], "min_samples_leaf": [1, 5, 20, 50, 200]}


def _evaluador(X, y):
    cv = StratifiedKFold(n_splits=u3.INNER_FOLDS, shuffle=True, random_state=SEED)
    pliegues = list(cv.split(X, y))
    pre = SkPipeline(u3.build_preprocessor_steps())

    # El preprocesamiento se ajusta dentro de cada pliegue (sin fuga): se precalcula una
    # vez por pliegue porque no depende de los hiperparámetros del árbol.
    cache = []
    for tr, va in pliegues:
        p = SkPipeline(u3.build_preprocessor_steps()).fit(X.iloc[tr], y.iloc[tr])
        cache.append((p.transform(X.iloc[tr]), y.iloc[tr].to_numpy(),
                      p.transform(X.iloc[va]), y.iloc[va].to_numpy()))
    del pre

    def evaluar(params):
        params = dict(params)
        if params.get("max_features") == "None":
            params["max_features"] = None
        s = []
        for Xtr, ytr, Xva, yva in cache:
            m = DecisionTreeClassifier(random_state=SEED, **params).fit(Xtr, ytr)
            s.append(f1_score(yva, m.predict(Xva), average="macro"))
        return float(np.mean(s))
    return evaluar


def _registrar(hist, metodo, semilla, params, score, t0):
    mejor = max([h["score"] for h in hist if h["metodo"] == metodo and h["semilla"] == semilla]
                + [score])
    n = sum(1 for h in hist if h["metodo"] == metodo and h["semilla"] == semilla) + 1
    hist.append({"metodo": metodo, "semilla": semilla, "evaluacion": n, "score": score,
                 "mejor_hasta_ahora": mejor, "tiempo_acumulado_s": time.perf_counter() - t0,
                 "params": json.dumps(params, default=str)})


def _muestra_aleatoria(rng):
    lo, hi = ESPACIO_DT["min_samples_leaf"]
    lo2, hi2 = ESPACIO_DT["min_samples_split"]
    return {
        "max_depth": int(rng.integers(ESPACIO_DT["max_depth"][0], ESPACIO_DT["max_depth"][1] + 1)),
        "min_samples_leaf": int(round(np.exp(rng.uniform(np.log(lo), np.log(hi))))),
        "min_samples_split": int(round(np.exp(rng.uniform(np.log(lo2), np.log(hi2))))),
        "criterion": str(rng.choice(ESPACIO_DT["criterion"])),
        "max_features": [None, "sqrt", 0.5][int(rng.integers(0, 3))],
    }


def anytime():
    import optuna
    from deap import base, creator, tools
    import random as pyrandom
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, *_ = u3.get_data()
    evaluar = _evaluador(X, y)
    hist = []

    # Grid Search: determinista; el orden de recorrido es el producto cartesiano.
    t0 = time.perf_counter()
    for md, msl in product(GRID_DT["max_depth"], GRID_DT["min_samples_leaf"]):
        p = {"max_depth": md, "min_samples_leaf": msl}
        _registrar(hist, "Grid Search", 0, p, evaluar(p), t0)
    print("grid ok", flush=True)

    for semilla in SEMILLAS:
        rng = np.random.default_rng(semilla)
        t0 = time.perf_counter()
        for _ in range(PRESUPUESTO):
            p = _muestra_aleatoria(rng)
            _registrar(hist, "Random Search", semilla, p, evaluar(p), t0)

        t0 = time.perf_counter()

        def objetivo(trial):
            p = {
                "max_depth": trial.suggest_int("max_depth", *ESPACIO_DT["max_depth"]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *ESPACIO_DT["min_samples_leaf"], log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", *ESPACIO_DT["min_samples_split"], log=True),
                "criterion": trial.suggest_categorical("criterion", ESPACIO_DT["criterion"]),
                "max_features": trial.suggest_categorical("max_features", ["None", "sqrt", 0.5]),
            }
            s = evaluar(p)
            _registrar(hist, "Bayesiana (TPE)", semilla, p, s, t0)
            return s
        estudio = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=8))
        estudio.optimize(objetivo, n_trials=PRESUPUESTO)

        # Genético: población 6, torneo de 3, cruce uniforme (0,6), mutación (0,3), elitismo 1.
        prng = pyrandom.Random(semilla)
        nrng = np.random.default_rng(semilla)
        if "FitAny" not in dir(creator):
            creator.create("FitAny", base.Fitness, weights=(1.0,))
            creator.create("IndAny", dict, fitness=creator.FitAny)
        t0 = time.perf_counter()
        evaluados = 0

        def ev(ind):
            nonlocal evaluados
            p = dict(ind)
            s = evaluar(p)
            evaluados += 1
            _registrar(hist, "Genético (DEAP)", semilla, p, s, t0)
            return (s,)

        pob = [creator.IndAny(_muestra_aleatoria(nrng)) for _ in range(6)]
        for ind in pob:
            ind.fitness.values = ev(ind)
        while evaluados < PRESUPUESTO:
            elite = tools.selBest(pob, 1)
            hijos = [creator.IndAny(dict(i)) for i in tools.selTournament(pob, len(pob) - 1, tournsize=3)]
            for h1, h2 in zip(hijos[::2], hijos[1::2]):
                if prng.random() < 0.6:
                    for k in ESPACIO_DT:
                        if prng.random() < 0.5:
                            h1[k], h2[k] = h2[k], h1[k]
            for h in hijos:
                if prng.random() < 0.3:
                    k = prng.choice(list(ESPACIO_DT))
                    h[k] = _muestra_aleatoria(nrng)[k]
            for h in hijos:
                if evaluados >= PRESUPUESTO:
                    h.fitness.values = (-1.0,)
                    continue
                h.fitness.values = ev(h)
            pob = [creator.IndAny(dict(e)) for e in elite] + hijos
            pob[0].fitness.values = elite[0].fitness.values
        print(f"semilla {semilla} ok", flush=True)

    df = pd.DataFrame(hist)
    df.to_csv(RESULTS / "anytime_historial.csv", index=False, encoding="utf-8-sig")
    resumen = (df.groupby(["metodo", "semilla"])
                 .agg(mejor_final=("score", "max"), tiempo_total_s=("tiempo_acumulado_s", "max"),
                      evaluaciones=("evaluacion", "max")).reset_index())
    resumen.to_csv(RESULTS / "anytime_resumen.csv", index=False, encoding="utf-8-sig")
    print(resumen.groupby("metodo")[["mejor_final", "tiempo_total_s"]].agg(["mean", "std"]))


# ---------------------------------------------------------------------------------------
# 2. Implementación estándar frente a optimizada
# ---------------------------------------------------------------------------------------

def _medir(fit, predict, Xte, yte, repeticiones=3):
    t_fit, t_pred, mem_fit = [], [], []
    for _ in range(repeticiones):
        tracemalloc.start()
        t0 = time.perf_counter()
        modelo = fit()
        t_fit.append(time.perf_counter() - t0)
        mem_fit.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
        t0 = time.perf_counter()
        pred = predict(modelo, Xte)
        t_pred.append(time.perf_counter() - t0)
    return {"tiempo_entrenamiento_s": float(np.median(t_fit)),
            "tiempo_inferencia_s": float(np.median(t_pred)),
            "memoria_pico_entrenamiento_MB": float(np.median(mem_fit)),
            "F1_macro": float(f1_score(yte, pred, average="macro"))}


def computacional():
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from xgboost import XGBClassifier

    X, y, X_test, y_test, *_ = u3.get_data()
    pre = SkPipeline(u3.build_preprocessor_steps()).fit(X, y)
    Xtr, ytr = pre.transform(X), y.to_numpy()
    _, Xte_df, _, yte_s = __import__("sklearn.model_selection", fromlist=["x"]).train_test_split(
        X_test, y_test, test_size=20_000, stratify=y_test, random_state=SEED)
    Xte, yte = pre.transform(Xte_df), yte_s.to_numpy()
    n_svm = u3.N_SVM
    filas = []

    def agregar(modelo, version, complejidad, fit, predict, n_train, rep=3):
        r = _medir(fit, predict, Xte, yte, repeticiones=rep)
        r.update({"modelo": modelo, "versión": version, "complejidad_teórica": complejidad,
                  "n_entrenamiento": n_train})
        filas.append(r)
        print(modelo, version, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

    # KNN: búsqueda exhaustiva frente a estructuras de indexación.
    for alg, cx in [("brute", "O(n·p) por consulta"), ("kd_tree", "O(log n) por consulta en baja dimensión; degrada a O(n·p) con p alta"),
                    ("ball_tree", "O(log n) por consulta en baja dimensión; más robusto que KD-Tree con p alta")]:
        agregar("KNN (k=5)", {"brute": "estándar: fuerza bruta"}.get(alg, f"optimizada: {alg}"), cx,
                lambda a=alg: KNeighborsClassifier(n_neighbors=5, algorithm=a, n_jobs=1).fit(Xtr, ytr),
                lambda m, Z: m.predict(Z), len(ytr), rep=1)

    # Naive Bayes: ajuste en bloque frente a aprendizaje incremental por lotes.
    def nb_incremental():
        m = GaussianNB()
        clases = np.unique(ytr)
        for i in range(0, len(ytr), 5_000):
            m.partial_fit(Xtr[i:i + 5_000], ytr[i:i + 5_000], classes=clases)
        return m
    agregar("Naive Bayes", "estándar: fit en bloque", "O(n·p)",
            lambda: GaussianNB().fit(Xtr, ytr), lambda m, Z: m.predict(Z), len(ytr))
    agregar("Naive Bayes", "optimizada: partial_fit (lotes de 5.000)", "O(n·p); memoria O(b·p) por lote",
            nb_incremental, lambda m, Z: m.predict(Z), len(ytr))

    # Regresión logística (L2): SAGA frente a L-BFGS.
    agregar("Regresión logística L2", "estándar: SAGA", "O(n·p) por época",
            lambda: LogisticRegression(solver="saga", max_iter=150, tol=1e-2, random_state=SEED).fit(Xtr, ytr),
            lambda m, Z: m.predict(Z), len(ytr))
    agregar("Regresión logística L2", "optimizada: L-BFGS", "O(n·p) por iteración; convergencia superlineal",
            lambda: LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED).fit(Xtr, ytr),
            lambda m, Z: m.predict(Z), len(ytr))

    # XGBoost: cortes exactos frente a histogramas y parada temprana.
    from sklearn.model_selection import train_test_split
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr - 1, test_size=0.15, stratify=ytr, random_state=SEED)
    comunes = dict(objective="multi:softprob", eval_metric="mlogloss", max_depth=8, learning_rate=0.2,
                   random_state=SEED, n_jobs=1)
    agregar("XGBoost", "estándar: exact, 200 árboles", "O(T·n·p·log n)",
            lambda: XGBClassifier(tree_method="exact", n_estimators=200, **comunes).fit(Xtr, ytr - 1),
            lambda m, Z: m.predict(Z) + 1, len(ytr), rep=1)
    agregar("XGBoost", "optimizada: hist, 200 árboles", "O(T·n·p) con 256 bins",
            lambda: XGBClassifier(tree_method="hist", n_estimators=200, **comunes).fit(Xtr, ytr - 1),
            lambda m, Z: m.predict(Z) + 1, len(ytr), rep=1)

    def xgb_es():
        m = XGBClassifier(tree_method="hist", n_estimators=1000, early_stopping_rounds=20, **comunes)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        return m
    agregar("XGBoost", "optimizada: hist + early stopping (validación 15 %)", "O(T*·n·p), T* ≤ T",
            xgb_es, lambda m, Z: m.predict(Z) + 1, len(ya), rep=1)

    # SVM: núcleo RBF frente a formulaciones lineales.
    idx = np.random.default_rng(SEED).choice(len(ytr), n_svm, replace=False)
    agregar("SVM", f"estándar: SVC RBF (n={n_svm:,})".replace(",", "."), "entre O(n²·p) y O(n³)",
            lambda: SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr[idx], ytr[idx]),
            lambda m, Z: m.predict(Z), n_svm, rep=1)
    agregar("SVM", "optimizada: LinearSVC", "O(n·p) por época (dual coordinate descent)",
            lambda: LinearSVC(C=1.0, random_state=SEED, max_iter=2000).fit(Xtr, ytr),
            lambda m, Z: m.predict(Z), len(ytr))
    agregar("SVM", "optimizada: SGDClassifier (hinge)", "O(n·p) por época",
            lambda: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-3, random_state=SEED).fit(Xtr, ytr),
            lambda m, Z: m.predict(Z), len(ytr))

    df = pd.DataFrame(filas)
    df = df[["modelo", "versión", "complejidad_teórica", "n_entrenamiento", "tiempo_entrenamiento_s",
             "tiempo_inferencia_s", "memoria_pico_entrenamiento_MB", "F1_macro"]]
    df.to_csv(RESULTS / "computacional_estandar_vs_optimizado.csv", index=False, encoding="utf-8-sig")
    with open(RESULTS / "computacional_entorno.json", "w", encoding="utf-8") as f:
        json.dump({**_entorno(), "n_prueba": int(len(yte))}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------------------
# 3. Exponente empírico de escalamiento
# ---------------------------------------------------------------------------------------

def escalamiento():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    X, y, X_test, y_test, X_full, y_full = u3.get_data(n=80_000)
    pre = SkPipeline(u3.build_preprocessor_steps()).fit(X, y)
    Xa, ya = pre.transform(X), y.to_numpy()
    Xq = pre.transform(X_test.iloc[:5_000])
    modelos = {
        "Naive Bayes": (lambda: GaussianNB(), [5_000, 10_000, 20_000, 40_000, 80_000], "fit"),
        "Regresión logística (L-BFGS)": (lambda: LogisticRegression(max_iter=1000), [5_000, 10_000, 20_000, 40_000, 80_000], "fit"),
        "Árbol de decisión": (lambda: DecisionTreeClassifier(random_state=SEED), [5_000, 10_000, 20_000, 40_000, 80_000], "fit"),
        "Random Forest (100 árboles)": (lambda: RandomForestClassifier(100, random_state=SEED, n_jobs=1), [5_000, 10_000, 20_000, 40_000], "fit"),
        "XGBoost hist (100 árboles)": (lambda: XGBClassifier(tree_method="hist", n_estimators=100, max_depth=6, n_jobs=1, random_state=SEED), [5_000, 10_000, 20_000, 40_000, 80_000], "fit"),
        "SVM RBF": (lambda: SVC(kernel="rbf"), [1_000, 2_000, 4_000, 8_000, 16_000], "fit"),
        "KNN (fuerza bruta, 5.000 consultas)": (lambda: KNeighborsClassifier(5, algorithm="brute", n_jobs=1), [5_000, 10_000, 20_000, 40_000, 80_000], "predict"),
    }
    filas = []
    for nombre, (mk, ns, fase) in modelos.items():
        for n in ns:
            yy = ya[:n] - (1 if "XGBoost" in nombre else 0)
            t0 = time.perf_counter()
            m = mk().fit(Xa[:n], yy)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            m.predict(Xq)
            t_pred = time.perf_counter() - t0
            filas.append({"modelo": nombre, "n": n, "fase_medida": fase,
                          "tiempo_s": t_fit if fase == "fit" else t_pred})
            print(nombre, n, round(filas[-1]["tiempo_s"], 3), flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(RESULTS / "escalamiento_tiempos.csv", index=False, encoding="utf-8-sig")
    exps = []
    for nombre, g in df.groupby("modelo", sort=False):
        b = np.polyfit(np.log(g["n"]), np.log(g["tiempo_s"]), 1)[0]
        exps.append({"modelo": nombre, "fase_medida": g["fase_medida"].iloc[0], "exponente_empirico": b})
    pd.DataFrame(exps).to_csv(RESULTS / "escalamiento_exponentes.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------------------
# 4. SHAP del finalista: resumen global y explicaciones locales
# ---------------------------------------------------------------------------------------

def shap_figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from xgboost import XGBClassifier

    hp = _hp_xgb()
    _, _, X_test, y_test, X_full, y_full = u3.get_data()
    pre = SkPipeline(u3.build_preprocessor_steps()).fit(X_full, y_full)
    nombres = list(u3.ENG_NUM_COLS) + list(u3.BIN_COLS)
    Xf = pre.transform(X_full)
    t0 = time.perf_counter()
    modelo = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                           random_state=SEED, n_jobs=-1, **hp).fit(Xf, y_full - 1)
    print(f"finalista reentrenado en {time.perf_counter() - t0:.0f} s", flush=True)
    Xt = pre.transform(X_test)
    pred = modelo.predict(Xt) + 1
    print("F1 macro prueba:", f1_score(y_test, pred, average="macro"), flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xt), 3_000, replace=False)
    explainer = shap.TreeExplainer(modelo)
    sv = explainer(pd.DataFrame(Xt[idx], columns=nombres))      # (n, p, clases)
    np.savez_compressed(RESULTS / "shap_valores_muestra.npz", valores=sv.values, base=sv.base_values,
                        datos=sv.data, nombres=np.array(nombres), indices=idx)

    figdir = RESULTS / "figuras"
    figdir.mkdir(exist_ok=True)
    clases = [u3.CLASS_SHORT[c] for c in range(1, 8)]

    # Importancia global apilada por clase.
    plt.figure()
    shap.summary_plot([sv.values[:, :, k] for k in range(7)], pd.DataFrame(Xt[idx], columns=nombres),
                      class_names=clases, max_display=15, show=False, plot_size=(9, 6))
    plt.title("Importancia global (|SHAP| medio) por clase")
    plt.tight_layout()
    plt.savefig(figdir / "shap_importancia_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Beeswarm para las dos clases mayoritarias, que concentran los errores.
    for k, nombre in [(0, "abeto_picea"), (1, "lodgepole")]:
        plt.figure()
        shap.plots.beeswarm(sv[:, :, k], max_display=15, show=False, plot_size=(9, 6))
        plt.title(f"Resumen SHAP para la clase {clases[k]}")
        plt.tight_layout()
        plt.savefig(figdir / f"shap_beeswarm_{nombre}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # Explicaciones locales: la observación correcta con mayor confianza y el error con
    # mayor confianza dentro de la muestra (el caso más informativo para el diagnóstico).
    proba = modelo.predict_proba(Xt[idx])
    pm = proba.argmax(1)
    real = y_test.to_numpy()[idx] - 1
    conf = proba.max(1)
    ok = np.where(pm == real)[0]
    err = np.where(pm != real)[0]
    i_ok = ok[np.argmax(conf[ok])]
    i_err = err[np.argmax(conf[err])]
    casos = {}
    for etiqueta, i in [("correcta", i_ok), ("error", i_err)]:
        k = int(pm[i])
        plt.figure()
        shap.plots.waterfall(sv[i, :, k], max_display=12, show=False)
        plt.title(f"Predicción {etiqueta}: real {clases[real[i]]}, predicha {clases[k]} (p = {conf[i]:.2f})",
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(figdir / f"shap_waterfall_{etiqueta}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        contrib = pd.Series(sv.values[i, :, k], index=nombres)
        casos[etiqueta] = {"indice_prueba": int(idx[i]), "clase_real": int(real[i] + 1),
                           "clase_predicha": int(k + 1), "probabilidad": float(conf[i]),
                           "probabilidad_clase_real": float(proba[i, real[i]]),
                           "top_contribuciones": contrib.reindex(contrib.abs().sort_values(ascending=False).index)
                           .head(8).round(4).to_dict()}
    with open(RESULTS / "shap_casos_waterfall.json", "w", encoding="utf-8") as f:
        json.dump(casos, f, ensure_ascii=False, indent=2)
    print(json.dumps(casos, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------
# 5. Generalización a áreas no vistas del finalista (misma prueba del capítulo 3)
# ---------------------------------------------------------------------------------------

def areas():
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from _utils import (build_logreg, load_covertype, split_covertype, stratified_sample,
                        wilderness_label)

    hp = _hp_xgb()
    _, X, y = load_covertype()
    X_train, _, y_train, _ = split_covertype(X, y)
    X_g, y_g = stratified_sample(X_train, y_train, 60000)   # misma submuestra del capítulo 3
    grupos = wilderness_label(X_g)
    filas = []
    for tr, te in GroupKFold(n_splits=4).split(X_g, y_g, grupos):
        area = grupos.iloc[te].iloc[0]
        ytr, yte = y_g.iloc[tr], y_g.iloc[te]
        # Configuración óptima de la sección 3.3 (ingeniería de variables, C = 10).
        lr = build_logreg(engineered=True, C=10.0, max_iter=3000).fit(X_g.iloc[tr], ytr)
        le = LabelEncoder().fit(ytr)
        xgb = SkPipeline(u3.build_preprocessor_steps() + [("clf", XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            random_state=SEED, n_jobs=-1, **hp))]).fit(X_g.iloc[tr], le.transform(ytr))
        for nombre, pred in [("Regresión logística", lr.predict(X_g.iloc[te])),
                             ("XGBoost", le.inverse_transform(xgb.predict(X_g.iloc[te])))]:
            filas.append({"modelo": nombre, "área_no_vista": area, "celdas": len(te),
                          "accuracy": accuracy_score(yte, pred),
                          "f1_macro_clases_presentes": f1_score(yte, pred, average="macro",
                                                               labels=sorted(yte.unique()), zero_division=0)})
        print(area, filas[-2:], flush=True)
    pd.DataFrame(filas).to_csv(RESULTS / "finalista_areas_no_vistas.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------------------
# 6. Figuras SHAP en español a partir de los valores guardados (sin reentrenar)
# ---------------------------------------------------------------------------------------
TRADUCCIONES = [
    ("mean(|SHAP value|) (average impact on model output magnitude)", "|SHAP| medio (impacto medio sobre el margen del modelo)"),
    ("SHAP value (impact on model output)", "Valor SHAP (impacto sobre el margen del modelo)"),
    ("Feature value", "Valor de la variable"),
    ("High", "Alto"), ("Low", "Bajo"),
    ("other features", "variables restantes"), ("Sum of", "Suma de"),
]


def _tr(s):
    for a, b in TRADUCCIONES:
        s = s.replace(a, b)
    return s


def _traducir(fig):
    import matplotlib.text as mtext
    fig.canvas.draw()
    for ax in fig.axes:
        for eje in (ax.yaxis, ax.xaxis):
            ticks = eje.get_ticklocs()
            etiquetas = [t.get_text() for t in eje.get_ticklabels()]
            if any(_tr(e) != e for e in etiquetas):
                eje.set_ticks(ticks)
                eje.set_ticklabels([_tr(e) for e in etiquetas])
    for t in fig.findobj(mtext.Text):
        t.set_text(_tr(t.get_text()))


def shap_figuras_es():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    z = np.load(RESULTS / "shap_valores_muestra.npz", allow_pickle=True)
    nombres = list(z["nombres"])
    exp = shap.Explanation(values=z["valores"], base_values=z["base"], data=z["datos"], feature_names=nombres)
    with open(RESULTS / "shap_casos_waterfall.json", encoding="utf-8") as f:
        casos = json.load(f)
    idx = list(z["indices"])
    clases = [u3.CLASS_SHORT[c] for c in range(1, 8)]
    figdir = RESULTS / "figuras"

    plt.figure()
    shap.summary_plot([exp.values[:, :, k] for k in range(7)], pd.DataFrame(exp.data, columns=nombres),
                      class_names=clases, max_display=15, show=False, plot_size=(9, 6))
    plt.title("Importancia global por clase")
    _traducir(plt.gcf())
    plt.savefig(figdir / "shap_importancia_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    for k, nombre in [(0, "abeto_picea"), (1, "lodgepole")]:
        plt.figure()
        shap.plots.beeswarm(exp[:, :, k], max_display=15, show=False, plot_size=(9, 6))
        plt.title(f"Resumen SHAP para la clase {clases[k]}")
        _traducir(plt.gcf())
        plt.savefig(figdir / f"shap_beeswarm_{nombre}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    for etiqueta, caso in casos.items():
        i = idx.index(caso["indice_prueba"])
        k = caso["clase_predicha"] - 1
        plt.figure()
        shap.plots.waterfall(exp[i, :, k], max_display=12, show=False)
        plt.title(f"Predicción {'correcta' if etiqueta == 'correcta' else 'errónea'}: real "
                  f"{clases[caso['clase_real'] - 1]}, predicha {clases[k]} (p = {caso['probabilidad']:.2f})",
                  fontsize=10)
        _traducir(plt.gcf())
        plt.savefig(figdir / f"shap_waterfall_{etiqueta}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    # Importancia global (|SHAP| medio sumado sobre clases) para la tabla del capítulo 8.
    imp = pd.Series(np.abs(exp.values).mean(axis=0).sum(axis=1), index=nombres).sort_values(ascending=False)
    imp.rename("importancia_media_abs").to_csv(RESULTS / "shap_importancia_global_muestra.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------------------------------
# 7. LIME sobre las mismas dos observaciones explicadas con SHAP (sección 8.2)
# ---------------------------------------------------------------------------------------

def lime_casos():
    from lime.lime_tabular import LimeTabularExplainer
    from xgboost import XGBClassifier

    hp = _hp_xgb()
    with open(RESULTS / "shap_casos_waterfall.json", encoding="utf-8") as f:
        casos = json.load(f)
    z = np.load(RESULTS / "shap_valores_muestra.npz", allow_pickle=True)
    nombres = list(z["nombres"])
    idx = list(z["indices"])
    _, _, X_test, y_test, X_full, y_full = u3.get_data()
    pre = SkPipeline(u3.build_preprocessor_steps()).fit(X_full, y_full)
    modelo = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                           random_state=SEED, n_jobs=-1, **hp).fit(pre.transform(X_full), y_full - 1)
    Xt = pre.transform(X_test)
    rng = np.random.default_rng(SEED)
    referencia = Xt[rng.choice(len(Xt), size=2000, replace=False)]
    explicador = LimeTabularExplainer(referencia, feature_names=nombres,
                                      class_names=[str(c) for c in range(1, 8)],
                                      discretize_continuous=True, random_state=SEED)
    salida = {}
    for etiqueta, caso in casos.items():
        k = caso["clase_predicha"] - 1
        e = explicador.explain_instance(Xt[caso["indice_prueba"]], modelo.predict_proba, num_features=10,
                                        labels=(k,), num_samples=5000)
        lime_pesos = {nombres[i]: float(w) for i, w in e.as_map()[k]}
        shap_vals = pd.Series(z["valores"][idx.index(caso["indice_prueba"]), :, k], index=nombres)
        top_shap = list(shap_vals.abs().sort_values(ascending=False).index[:5])
        top_lime = list(pd.Series(lime_pesos).abs().sort_values(ascending=False).index[:5])
        salida[etiqueta] = {**caso, "lime_pesos": lime_pesos, "lime_r2_local": float(e.score),
                            "top5_shap": top_shap, "top5_lime": top_lime,
                            "comunes_top5": sorted(set(top_shap) & set(top_lime))}
        print(etiqueta, salida[etiqueta]["comunes_top5"], round(e.score, 3), flush=True)
    with open(RESULTS / "shap_lime_casos.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    tareas = {"anytime": anytime, "computacional": computacional, "escalamiento": escalamiento,
              "shap": shap_figuras, "areas": areas,
              "figuras_es": shap_figuras_es, "lime": lime_casos}
    for t in sys.argv[1:]:
        print(f"=== {t} ===", flush=True)
        tareas[t]()
