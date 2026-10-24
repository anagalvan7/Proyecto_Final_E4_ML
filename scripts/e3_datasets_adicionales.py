"""Evaluacion del ECLC en tres conjuntos de datos adicionales de clasificacion.

Conjuntos incluidos en scikit-learn (no requieren descarga, reproducibles sin conexion):
    digits         1.797 x 64, 10 clases   imagenes de digitos manuscritos
    wine             178 x 13,  3 clases   analisis quimico de vinos
    breast_cancer    569 x 30,  2 clases   variables clinicas de imagenes de biopsia

Protocolo identico para todos los modelos: validacion cruzada repetida 5 x 3
estratificada (semilla 42 + r); los 7 modelos base y HistGradientBoosting se optimizan
con Random Search de 6 configuraciones en 3 pliegues internos (el presupuesto del
benchmark); el ECLC optimiza (k, beta, tau) con 6 ensayos de TPE sobre las
probabilidades fuera de pliegue. Metrica: F1 macro.

Salida: resultados/e3/datasets_adicionales.csv (una fila por conjunto, pliegue y metodo).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from src.original_model import EnsambleCompetenciaLocal, vecinos_mas_cercanos  # noqa: E402

SEMILLA = 42
N_REP, N_PLIEGUES, N_ITER = 5, 3, 6
SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "e3" / "datasets_adicionales.csv"

CONJUNTOS = {"digits": load_digits, "wine": load_wine, "breast_cancer": load_breast_cancer}

# Mismos espacios de busqueda del benchmark de Covertype (_utils3.MODELOS), con escalado
# estandar como unico preprocesamiento.
MODELOS = {
    "KNN": (lambda **p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
            {"n_neighbors": [5, 11, 21, 41], "weights": ["uniform", "distance"]}),
    "NB": (lambda **p: GaussianNB(**p), {"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}),
    "LR": (lambda **p: make_pipeline(StandardScaler(), LogisticRegression(
        solver="saga", max_iter=2000, random_state=SEMILLA, **p)),
        {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}),
    "DT": (lambda **p: DecisionTreeClassifier(random_state=SEMILLA, **p),
           {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 10, 50]}),
    "RF": (lambda **p: RandomForestClassifier(random_state=SEMILLA, n_jobs=1, **p),
           {"n_estimators": [100, 200], "max_depth": [10, 20]}),
    "XGB": (lambda **p: XGBClassifier(tree_method="hist", random_state=SEMILLA, n_jobs=1, **p),
            {"n_estimators": [100, 150], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}),
    "SVM": (lambda **p: make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMILLA, **p)),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]}),
}
HGB = (lambda **p: HistGradientBoostingClassifier(random_state=SEMILLA, early_stopping=False, **p),
       {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31, 63, 127], "max_iter": [100, 200],
        "l2_regularization": [0.0, 1.0]})


def f1m(y, pred):
    return f1_score(y, pred, average="macro")


def random_search(construir, espacio, X, y):
    rng = np.random.default_rng(SEMILLA)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEMILLA)
    mejor, mejor_s = None, -1.0
    for _ in range(N_ITER):
        p = {k: v[rng.integers(len(v))] for k, v in espacio.items()}
        s = np.mean([f1m(y[b], construir(**p).fit(X[a], y[a]).predict(X[b])) for a, b in cv.split(X, y)])
        if s > mejor_s:
            mejor, mejor_s = p, s
    return construir(**mejor)


def oof(est, X, y, C):
    P = np.zeros((len(X), C))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEMILLA).split(X, y):
        P[b] = clone(est).fit(X[a], y[a]).predict_proba(X[b])
    return P


def optimizar_eclc(X, y, P_oof):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEMILLA)

    def objetivo(t):
        hp = {"k": t.suggest_int("k", 5, min(200, int(len(X) * 2 / 3) - 1), log=True),
              "beta": t.suggest_float("beta", 0.1, 50.0, log=True),
              "tau": t.suggest_float("tau", 0.1, 1e4, log=True)}
        s = []
        for a, b in cv.split(X, y):
            m = EnsambleCompetenciaLocal([("b", None)], **hp).ajustar_desde_oof(X[a], y[a], P_oof[:, a])
            s.append(f1m(y[b], m.combinar(P_oof[:, b], X=X[b]).argmax(1)))
        return float(np.mean(s))

    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    est.optimize(objetivo, n_trials=N_ITER)
    return est.best_params


def un_pliegue(nombre, rep, f, X, y, tr, te):
    C = len(np.unique(y))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    P_oof, P_te, filas = [], [], []
    for m, (construir, espacio) in MODELOS.items():
        est = random_search(construir, espacio, Xtr, ytr)
        P_oof.append(oof(est, Xtr, ytr, C))
        P_te.append(clone(est).fit(Xtr, ytr).predict_proba(Xte))
        filas.append((f"base:{m}", f1m(yte, P_te[-1].argmax(1))))
    P_oof, P_te = np.stack(P_oof), np.stack(P_te)
    hgb = random_search(*HGB, Xtr, ytr).fit(Xtr, ytr)
    filas.append(("HGB (estado del arte)", f1m(yte, hgb.predict(Xte))))
    hp = optimizar_eclc(Xtr, ytr, P_oof)
    modelo = EnsambleCompetenciaLocal([("b", None)], **hp).ajustar_desde_oof(Xtr, ytr, P_oof)
    filas.append(("ECLC", f1m(yte, modelo.combinar(P_te, X=Xte).argmax(1))))
    filas.append(("promedio simple (beta = 0)", f1m(yte, P_te.mean(0).argmax(1))))
    lr = LogisticRegression(max_iter=2000, random_state=SEMILLA).fit(
        np.log(np.clip(P_oof, 1e-6, 1)).transpose(1, 0, 2).reshape(len(tr), -1), ytr)
    filas.append(("stacking lineal", f1m(yte, lr.predict(
        np.log(np.clip(P_te, 1e-6, 1)).transpose(1, 0, 2).reshape(len(te), -1)))))
    # KNORA-E con 7 vecinos en el mismo espacio estandarizado del ECLC
    Zr = ((Xtr - modelo.media_) / modelo.escala_).astype(np.float32)
    Zt = ((Xte - modelo.media_) / modelo.escala_).astype(np.float32)
    idx, _ = vecinos_mas_cercanos(Zr, Zt, 7)
    acierto = (P_oof.argmax(2) == ytr[None, :]).T[idx]
    todos = np.cumprod(acierto, axis=1).astype(bool)
    hay = todos.any(2)
    sel = np.ones((len(te), P_te.shape[0]), bool)
    ok = hay.any(1)
    t_max = 6 - np.argmax(hay[:, ::-1], axis=1)
    sel[ok] = todos[np.where(ok)[0], t_max[ok]]
    filas.append(("KNORA-E", f1m(yte, np.einsum("qm,mqc->qc", sel.astype(float), P_te).argmax(1))))
    return [{"conjunto": nombre, "rep": rep, "pliegue": f, "metodo": m, "f1_macro": s,
             "hiperparametros_eclc": json.dumps(hp)} for m, s in filas]


def main():
    """Un proceso por conjunto (--conjunto nombre) y los pliegues en secuencia: XGBoost
    falla con violaciones de acceso dentro de los trabajadores de joblib en Windows.
    Sin argumentos, solo une los CSV parciales en datasets_adicionales.csv."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=list(CONJUNTOS))
    args = ap.parse_args()
    if args.conjunto:
        X, y = CONJUNTOS[args.conjunto](return_X_y=True)
        X = X.astype(np.float64)
        filas = []
        for rep in range(N_REP):
            cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA + rep)
            for f, (tr, te) in enumerate(cv.split(X, y)):
                filas += un_pliegue(args.conjunto, rep, f, X, y, tr, te)
                print(args.conjunto, rep, f, flush=True)
        pd.DataFrame(filas).to_csv(SALIDA.with_name(f"datasets_adicionales_{args.conjunto}.csv"),
                                   index=False, encoding="utf-8-sig")
        return
    df = pd.concat([pd.read_csv(SALIDA.with_name(f"datasets_adicionales_{c}.csv"), encoding="utf-8-sig")
                    for c in CONJUNTOS])
    df.to_csv(SALIDA, index=False, encoding="utf-8-sig")
    print(df.groupby(["conjunto", "metodo"])["f1_macro"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
