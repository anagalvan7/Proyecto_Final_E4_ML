"""Capa de combinación en el protocolo de datos completos (ver e3c_probas_completo.py).

Para cada pliegue (o la prueba reservada) evalúa, sobre las mismas probabilidades: el ECLC con
(k, beta, tau) optimizados por TPE con 6 ensayos en validación interna de 3 pliegues (el mismo
presupuesto del benchmark), sus variantes de ablación y las reglas estándar (promedio, pesos por
error global, stacking lineal, KNORA-E, OLA), los 7 modelos base y HistGradientBoosting.

Con ~370.000 celdas de referencia, la búsqueda exacta de vecinos domina el costo. Los vecinos no
dependen de beta ni de tau, y para k basta con tomar los primeros k de una búsqueda con
K_MAX = 200 (el límite superior del espacio de k). Por eso se busca una sola vez por partición
(3 internas + 1 de prueba) y todas las configuraciones y variantes reutilizan esa búsqueda
(EnsambleCompetenciaLocal.combinar(..., vecinos=...)). El resultado es idéntico al de buscar de nuevo,
salvo empates exactos de distancia en el k-ésimo vecino (tests/test_original_model.py).

Uso: python scripts/e3c_meta.py --rep 0 --pliegue 3      |   python scripts/e3c_meta.py --holdout
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

import _e3 as e3
import _utils3 as u3
import e3_meta as em
from src.original_model import vecinos_mas_cercanos

SALIDA = e3.RES_E3 / "completo"
K_MAX = em.ESPACIO["k"][1]
TAM_BLOQUE = 256


def eclc(**hp):
    m = em.eclc(**hp)
    m.tam_bloque = TAM_BLOQUE
    return m


def buscar(Z_ref, Z_q):
    t0 = time.perf_counter()
    v = vecinos_mas_cercanos(Z_ref, Z_q, K_MAX, TAM_BLOQUE)
    print(f"   búsqueda de vecinos {len(Z_q)} x {len(Z_ref)}: {time.perf_counter() - t0:.0f}s", flush=True)
    return v


def optimizar(X, y, P_oof, semilla=u3.RANDOM_STATE):
    """Igual que e3_meta.optimizar (TPE, 6 ensayos, 3 pliegues internos), con vecinos reutilizados."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    particiones = []
    for a, b in StratifiedKFold(3, shuffle=True, random_state=semilla).split(X, y):
        ref = eclc(k=5, beta=1.0, tau=1.0).ajustar_desde_oof(X.iloc[a], y[a], P_oof[:, a])
        Zb = ref.transformador_vecinos_.transform(X.iloc[b]).astype(np.float32)
        particiones.append((a, b, buscar(ref.Z_ref_, Zb)))
    historia = []

    def objetivo(trial):
        hp = {"k": trial.suggest_int("k", *em.ESPACIO["k"], log=True),
              "beta": trial.suggest_float("beta", *em.ESPACIO["beta"], log=True),
              "tau": trial.suggest_float("tau", *em.ESPACIO["tau"], log=True)}
        s = []
        for a, b, vec in particiones:
            m = eclc(**hp).ajustar_desde_oof(X.iloc[a], y[a], P_oof[:, a])
            s.append(em.f1m(y[b], m.combinar(P_oof[:, b], vecinos=vec)))
        historia.append({**hp, "f1_cv": float(np.mean(s))})
        return float(np.mean(s))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=em.N_ENSAYOS)
    return dict(estudio.best_params), historia


def seleccion_dinamica(idx7, y_ref, P_oof, P_te, k=em.K_DES):
    """KNORA-E y OLA (misma lógica que e3_meta.seleccion_dinamica) con los primeros k vecinos ya buscados."""
    acierto = (e3.CLASES[P_oof.argmax(2)] == y_ref[None, :]).T
    a_vec = acierto[idx7[:, :k]]
    todos = np.cumprod(a_vec, axis=1).astype(bool)
    hay = todos.any(axis=2)
    t_max = np.where(hay.any(1), k - 1 - np.argmax(hay[:, ::-1], axis=1), -1)
    sel = np.ones((len(idx7), P_te.shape[0]), dtype=bool)
    ok = t_max >= 0
    sel[ok] = todos[np.where(ok)[0], t_max[ok]]
    P_knora = np.einsum("qm,mqc->qc", sel.astype(float), P_te) / sel.sum(1, keepdims=True)
    elegido = (a_vec.mean(axis=1) + 1e-9 * acierto.mean(0)[None, :]).argmax(1)
    return P_knora, P_te[elegido, np.arange(len(idx7))]


def evaluar(nombre, X_tr, y_tr, X_te, y_te, P_oof, P_te, P_hgb, modelos):
    t0 = time.perf_counter()
    hp, historia = optimizar(X_tr, y_tr, P_oof)
    t_opt = time.perf_counter() - t0
    modelo = eclc(**hp).ajustar_desde_oof(X_tr, y_tr, P_oof)
    Zte = modelo.transformador_vecinos_.transform(X_te).astype(np.float32)
    vec = buscar(modelo.Z_ref_, Zte)
    t0 = time.perf_counter()
    probas = {"ECLC": modelo.combinar(P_te, vecinos=vec)}
    t_inf = time.perf_counter() - t0
    ablacion = {"sin pesos por clase": {"por_clase": False}, "sin vecindario (tau = inf)": {"tau": np.inf},
                "sin contraccion (tau = 0)": {"tau": 0.0}, "sin nucleo de distancia": {"ponderar_distancia": False},
                "sin balanceo de vecinos": {"balancear_clases": False}, "promedio simple (beta = 0)": {"beta": 0.0}}
    for n, cambio in ablacion.items():
        probas[n] = eclc(**{**hp, **cambio}).ajustar_desde_oof(X_tr, y_tr, P_oof).combinar(P_te, vecinos=vec)
    w = em.pesos_error_global(y_tr, P_oof)
    probas["pesos por error global"] = np.einsum("m,mqc->qc", w, P_te)
    probas["stacking lineal"] = em.stacking_lineal(y_tr, P_oof, P_te)
    probas["KNORA-E"], probas["OLA"] = seleccion_dinamica(vec[0], y_tr, P_oof, P_te)
    for i, m in enumerate(modelos):
        probas[f"base:{m}"] = P_te[i]
    probas["HGB (estado del arte)"] = P_hgb
    metricas = {}
    for n, P in probas.items():
        P = P / P.sum(1, keepdims=True)
        pred = e3.CLASES[P.argmax(1)]
        metricas[n] = {"f1_macro": f1_score(y_te, pred, average="macro"), "accuracy": accuracy_score(y_te, pred),
                       "log_loss": log_loss(y_te, np.clip(P, 1e-15, 1), labels=e3.CLASES),
                       "f1_por_clase": f1_score(y_te, pred, average=None, labels=e3.CLASES).tolist()}
    salida = {"nombre": nombre, "hiperparametros": hp, "historia_optuna": historia, "tiempo_optimizacion_s": t_opt,
              "tiempo_inferencia_meta_s": t_inf, "n_prueba": int(len(y_te)), "n_entrenamiento": int(len(y_tr)),
              "modelos": modelos, "pesos_medios": modelo.pesos(vecinos=vec).mean(0).tolist(), "metricas": metricas}
    json.dump(salida, open(SALIDA / f"meta_{nombre}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(SALIDA / f"pred_{nombre}.npz", y_te=y_te,
                        **{k.replace(" ", "_").replace(":", "_"): e3.CLASES[v.argmax(1)] for k, v in probas.items()})
    print(f"{nombre}: hp={hp} ECLC={metricas['ECLC']['f1_macro']:.4f} "
          f"KNN={metricas['base:KNN']['f1_macro']:.4f} DT={metricas['base:DT']['f1_macro']:.4f} "
          f"stacking={metricas['stacking lineal']['f1_macro']:.4f} HGB={metricas['HGB (estado del arte)']['f1_macro']:.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int)
    ap.add_argument("--pliegue", type=int)
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    _, _, X_test, y_test, X_train, y_train = u3.get_data(n=u3.N_BENCHMARK)
    if a.holdout:
        d = np.load(SALIDA / "holdout.npz")
        X_tr, X_te, nombre = X_train, X_test, "holdout"
    else:
        d = np.load(SALIDA / f"rep{a.rep}_f{a.pliegue}.npz")
        idx = np.load(SALIDA / f"rep{a.rep}_f{a.pliegue}_idx.npz")
        X_tr, X_te, nombre = X_train.iloc[idx["tr"]], X_train.iloc[idx["te"]], f"rep{a.rep}_f{a.pliegue}"
    evaluar(nombre, X_tr, d["y_tr"], X_te, d["y_te"], d["P_oof"].astype(np.float64), d["P_te"].astype(np.float64),
            d["P_hgb"].astype(np.float64), [str(m) for m in d["modelos"]])


if __name__ == "__main__":
    main()
