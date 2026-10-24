"""Probabilidades de los 7 modelos base en un pliegue externo (repeticion r, pliegue f).

Para el pliegue externo (entrenamiento E, prueba T) de la validacion cruzada repetida:
1. Cada modelo base, con su mejor combinacion (balanceo + hiperparametros del benchmark),
   produce probabilidades fuera de pliegue sobre E (3 pliegues internos): forman el
   conjunto de referencia del ECLC, del stacking y de las demas reglas de combinacion.
2. Cada modelo base se reentrena con todo E y predice T.
3. Un competidor adicional del estado del arte, HistGradientBoosting (Ke et al., 2017),
   se optimiza con Random Search de 6 configuraciones en 3 pliegues internos (el mismo
   presupuesto de Random Search de los modelos del benchmark) y predice T.

Todo queda en resultados/e3/probas/rep<r>_f<f>.npz. Ningun hiperparametro del ECLC se usa
aqui: la capa de combinacion se ajusta despues (e3_meta.py) sobre estas probabilidades.

Uso: python scripts/e3_probas_pliegue.py --rep 0 --pliegue 1
"""
import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import _e3 as e3
import _utils3 as u3

ESPACIO_HGB = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [31, 63, 127],
    "max_iter": [100, 200],
    "l2_regularization": [0.0, 1.0],
}


def hgb_random_search(X, y, n_iter=6):
    rng = np.random.default_rng(u3.RANDOM_STATE)
    cv = StratifiedKFold(3, shuffle=True, random_state=u3.RANDOM_STATE)
    mejor, mejor_s = None, -1
    for _ in range(n_iter):
        p = {k: v[rng.integers(len(v))] for k, v in ESPACIO_HGB.items()}
        s = []
        for tr, va in cv.split(X, y):
            m = HistGradientBoostingClassifier(random_state=u3.RANDOM_STATE, early_stopping=False, **p)
            m.fit(X.iloc[tr], y.iloc[tr])
            s.append(f1_score(y.iloc[va], m.predict(X.iloc[va]), average="macro"))
        if np.mean(s) > mejor_s:
            mejor, mejor_s = p, float(np.mean(s))
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--pliegue", type=int, required=True)
    args = ap.parse_args()
    salida = e3.RES_E3 / "probas" / f"rep{args.rep}_f{args.pliegue}.npz"
    salida.parent.mkdir(exist_ok=True)
    if salida.exists():
        print(f"{salida.name}: ya existe")
        return

    X, y, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    tr, te = e3.particion(args.rep, X, y)[args.pliegue]
    X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    mejores = e3.mejores_combinaciones()

    M, C = len(mejores), len(e3.CLASES)
    P_oof = np.zeros((M, len(tr), C))
    P_te = np.zeros((M, len(te), C))
    tiempos = {}
    params_usados = {}
    for i, fila in mejores.iterrows():
        m, b = fila["modelo"], fila["balanceo"]
        params = e3.params_para(fila, args.rep, args.pliegue)
        params_usados[m] = params
        t0 = time.perf_counter()
        P_oof[i] = e3.probas_oof(m, b, params, X_tr, y_tr)
        t1 = time.perf_counter()
        pipe = e3.ajustar_modelo(m, b, params, X_tr, y_tr)
        t2 = time.perf_counter()
        P_te[i] = e3.proba_ordenada(pipe, m, X_te)
        t3 = time.perf_counter()
        tiempos[m] = {"oof_s": t1 - t0, "ajuste_s": t2 - t1, "inferencia_s": t3 - t2}
        f1 = f1_score(y_te, e3.CLASES[P_te[i].argmax(1)], average="macro")
        print(f"rep{args.rep} f{args.pliegue} {m}|{b}: F1={f1:.4f} ({t3 - t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    p_hgb = hgb_random_search(X_tr, y_tr)
    hgb = HistGradientBoostingClassifier(random_state=u3.RANDOM_STATE, early_stopping=False, **p_hgb)
    hgb.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    P_hgb = hgb.predict_proba(X_te)
    tiempos["HGB"] = {"busqueda_y_ajuste_s": t1 - t0, "inferencia_s": time.perf_counter() - t1}
    print(f"rep{args.rep} f{args.pliegue} HGB: F1={f1_score(y_te, hgb.predict(X_te), average='macro'):.4f}",
          flush=True)

    np.savez_compressed(
        salida, idx_tr=tr, idx_te=te, y_tr=y_tr.to_numpy(), y_te=y_te.to_numpy(),
        P_oof=P_oof.astype(np.float32), P_te=P_te.astype(np.float32), P_hgb=P_hgb.astype(np.float32),
        modelos=np.array(mejores["modelo"].tolist()),
        meta=json.dumps({"tiempos": tiempos, "params": params_usados, "params_hgb": p_hgb,
                         "balanceo": dict(zip(mejores["modelo"], mejores["balanceo"])),
                         "optimizacion": dict(zip(mejores["modelo"], mejores["optimizacion"]))}),
    )


if __name__ == "__main__":
    main()
