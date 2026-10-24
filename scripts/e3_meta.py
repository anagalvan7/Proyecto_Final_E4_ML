"""Capa de combinacion en cada pliegue externo: ECLC, ablacion y reglas de referencia.

Lee resultados/e3/probas/rep<r>_f<f>.npz (e3_probas_pliegue.py) y, sin reentrenar los
modelos base:
1. Optimiza los hiperparametros del ECLC (k, beta, tau) con optimizacion bayesiana (Optuna,
   TPE) y el mismo presupuesto del benchmark (6 ensayos), con validacion cruzada interna
   de 3 pliegues sobre las probabilidades fuera de pliegue del entrenamiento externo.
2. Evalua en la prueba externa el ECLC y sus variantes de ablacion (cada componente
   retirado por separado) con los mismos hiperparametros.
3. Evalua reglas de combinacion estandar sobre las mismas probabilidades: promedio simple,
   pesos por error global, stacking lineal, KNORA-E y OLA (seleccion dinamica, propias).

Guarda resultados/e3/meta/rep<r>_f<f>.json y las predicciones en .npz.
Uso: python scripts/e3_meta.py --rep 0 --pliegue 0     (o --todos)
"""
import argparse
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

import _e3 as e3
import _utils3 as u3
from src.original_model import EnsambleCompetenciaLocal, vecinos_mas_cercanos

N_ENSAYOS = 6
# Iteracion 2 del diseno: tau llega hasta 1e4 para que la optimizacion pueda elegir el
# regimen casi global (tau -> inf) ademas del local (capitulo 5, ciclo de desarrollo).
ESPACIO = {"k": (5, 200), "beta": (0.1, 50.0), "tau": (0.1, 1e4)}
K_DES = 7  # vecindario estandar de KNORA-E y OLA (Ko et al., 2008; Woods et al., 1997)


def eclc(**hp):
    return EnsambleCompetenciaLocal(estimadores=[("base", None)], random_state=u3.RANDOM_STATE,
                                    transformador_vecinos=_Prep(), **hp)


class _Prep:
    """Envoltorio clonable del preprocesamiento del benchmark (espacio de vecinos)."""

    def get_params(self, deep=False):
        return {}

    def fit(self, X):
        self.p_ = e3.espacio_vecinos(X)
        return self

    def transform(self, X):
        return self.p_.transform(X)


def f1m(y, P):
    return f1_score(y, e3.CLASES[P.argmax(1)], average="macro")


def cv_meta(X, y, P_oof, hp, semilla=u3.RANDOM_STATE):
    """F1 macro del ECLC en validacion cruzada interna sobre las probabilidades OOF."""
    cv = StratifiedKFold(3, shuffle=True, random_state=semilla)
    s = []
    for a, b in cv.split(X, y):
        m = eclc(**hp).ajustar_desde_oof(X.iloc[a], y[a], P_oof[:, a])
        s.append(f1m(y[b], m.combinar(P_oof[:, b], X=X.iloc[b])))
    return float(np.mean(s))


def optimizar(X, y, P_oof, n_ensayos=N_ENSAYOS, semilla=u3.RANDOM_STATE):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    historia = []

    def objetivo(trial):
        hp = {"k": trial.suggest_int("k", *ESPACIO["k"], log=True),
              "beta": trial.suggest_float("beta", *ESPACIO["beta"], log=True),
              "tau": trial.suggest_float("tau", *ESPACIO["tau"], log=True)}
        s = cv_meta(X, y, P_oof, hp, semilla)
        historia.append({**hp, "f1_cv": s})
        return s

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=n_ensayos)
    return dict(estudio.best_params), historia


# ---------------------------------------------------------------- reglas de referencia
def pesos_error_global(y, P_oof):
    """w_m proporcional a 1 / (1 - F1 macro OOF del modelo m)."""
    err = np.array([1 - f1m(y, P) for P in P_oof])
    w = 1 / np.maximum(err, 1e-6)
    return w / w.sum()


def stacking_lineal(y, P_oof, P_te):
    """Regresion logistica multinomial sobre log-probabilidades OOF (Wolpert, 1992)."""
    F_tr = np.log(np.clip(P_oof, 1e-6, 1)).transpose(1, 0, 2).reshape(P_oof.shape[1], -1)
    F_te = np.log(np.clip(P_te, 1e-6, 1)).transpose(1, 0, 2).reshape(P_te.shape[1], -1)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=u3.RANDOM_STATE).fit(F_tr, y)
    P = np.zeros((P_te.shape[1], len(e3.CLASES)))
    P[:, np.searchsorted(e3.CLASES, lr.classes_)] = lr.predict_proba(F_te)
    return P


def seleccion_dinamica(Z_ref, y_ref, P_oof, Z_te, P_te, k=K_DES):
    """KNORA-E y OLA con la misma busqueda de vecinos del ECLC.

    KNORA-Eliminate: conserva los modelos que aciertan en los k vecinos; si ninguno lo
    logra, reduce el vecindario hasta que alguno acierte (si ni con 1 vecino, usa todos) y
    promedia los seleccionados. OLA: elige el modelo con mayor exactitud local.
    """
    idx, _ = vecinos_mas_cercanos(Z_ref, Z_te, k)
    acierto = (e3.CLASES[P_oof.argmax(2)] == y_ref[None, :]).T           # (n_ref, M)
    a_vec = acierto[idx]                                                  # (n_te, k, M)
    todos = np.cumprod(a_vec, axis=1).astype(bool)                        # acierta en los t primeros
    hay = todos.any(axis=2)                                               # (n_te, k)
    t_max = np.where(hay.any(1), k - 1 - np.argmax(hay[:, ::-1], axis=1), -1)
    sel = np.ones((len(Z_te), P_te.shape[0]), dtype=bool)
    ok = t_max >= 0
    sel[ok] = todos[np.where(ok)[0], t_max[ok]]
    P_knora = np.einsum("qm,mqc->qc", sel.astype(float), P_te) / sel.sum(1, keepdims=True)
    exact_local = a_vec.mean(axis=1) + 1e-9 * acierto.mean(0)[None, :]    # desempate: exactitud global
    elegido = exact_local.argmax(1)
    P_ola = P_te[elegido, np.arange(len(Z_te))]
    return P_knora, P_ola


def evaluar(rep, pliegue):
    datos = np.load(e3.RES_E3 / "probas" / f"rep{rep}_f{pliegue}.npz", allow_pickle=False)
    modelos = [str(m) for m in datos["modelos"]]
    X, y_all, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    X_tr, X_te = X.iloc[datos["idx_tr"]], X.iloc[datos["idx_te"]]
    y_tr, y_te = datos["y_tr"], datos["y_te"]
    P_oof = datos["P_oof"].astype(np.float64)
    P_te = datos["P_te"].astype(np.float64)

    t0 = time.perf_counter()
    hp, historia = optimizar(X_tr, y_tr, P_oof)
    t_opt = time.perf_counter() - t0
    t0 = time.perf_counter()
    modelo = eclc(**hp).ajustar_desde_oof(X_tr, y_tr, P_oof)
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    P_eclc = modelo.combinar(P_te, X=X_te)
    t_inf = time.perf_counter() - t0

    probas = {"ECLC": P_eclc}
    ablacion = {
        "sin pesos por clase": {"por_clase": False},
        "sin vecindario (tau = inf)": {"tau": np.inf},
        "sin contraccion (tau = 0)": {"tau": 0.0},
        "sin nucleo de distancia": {"ponderar_distancia": False},
        "sin balanceo de vecinos": {"balancear_clases": False},
        "promedio simple (beta = 0)": {"beta": 0.0},
    }
    for nombre, cambio in ablacion.items():
        m = eclc(**{**hp, **cambio}).ajustar_desde_oof(X_tr, y_tr, P_oof)
        probas[nombre] = m.combinar(P_te, X=X_te)
    w = pesos_error_global(y_tr, P_oof)
    probas["pesos por error global"] = np.einsum("m,mqc->qc", w, P_te)
    probas["stacking lineal"] = stacking_lineal(y_tr, P_oof, P_te)
    probas["KNORA-E"], probas["OLA"] = seleccion_dinamica(
        modelo.Z_ref_, y_tr, P_oof, modelo.transformador_vecinos_.transform(X_te).astype(np.float32), P_te)
    for i, m in enumerate(modelos):
        probas[f"base:{m}"] = P_te[i]
    probas["HGB (estado del arte)"] = datos["P_hgb"].astype(np.float64)

    metricas = {}
    for nombre, P in probas.items():
        P = P / P.sum(1, keepdims=True)
        metricas[nombre] = {"f1_macro": f1m(y_te, P),
                            "accuracy": accuracy_score(y_te, e3.CLASES[P.argmax(1)]),
                            "log_loss": log_loss(y_te, np.clip(P, 1e-15, 1), labels=e3.CLASES),
                            "f1_por_clase": f1_score(y_te, e3.CLASES[P.argmax(1)], average=None,
                                                     labels=e3.CLASES).tolist()}
    W = modelo.pesos(X=X_te)                                      # (n, M, C)
    salida = {"rep": rep, "pliegue": pliegue, "hiperparametros": hp, "historia_optuna": historia,
              "tiempo_optimizacion_s": t_opt, "tiempo_ajuste_meta_s": t_ajuste,
              "tiempo_inferencia_meta_s": t_inf, "n_prueba": int(len(y_te)),
              "pesos_medios": W.mean(0).tolist(), "modelos": modelos,
              "pesos_error_global": w.tolist(), "metricas": metricas,
              "meta_probas": json.loads(str(datos["meta"]))}
    destino = e3.RES_E3 / "meta"
    destino.mkdir(exist_ok=True)
    with open(destino / f"rep{rep}_f{pliegue}.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)
    np.savez_compressed(destino / f"rep{rep}_f{pliegue}_pred.npz", y_te=y_te, idx_te=datos["idx_te"],
                        **{k.replace(" ", "_").replace(":", "_"): e3.CLASES[v.argmax(1)] for k, v in probas.items()})
    print(f"rep{rep} f{pliegue}: hp={hp} ECLC={metricas['ECLC']['f1_macro']:.4f} "
          f"mejor base={max(metricas[f'base:{m}']['f1_macro'] for m in modelos):.4f} "
          f"stacking={metricas['stacking lineal']['f1_macro']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int)
    ap.add_argument("--pliegue", type=int)
    args = ap.parse_args()
    evaluar(args.rep, args.pliegue)


if __name__ == "__main__":
    main()
