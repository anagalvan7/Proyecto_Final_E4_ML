"""Protocolo de datos completos: probabilidades de los 7 benchmarks con el entrenamiento completo.

Mismo diseño que e3_probas_pliegue.py, pero sobre las 464.809 celdas de entrenamiento en lugar de
la submuestra de 40.000. Validación cruzada externa 5 × 2 (StratifiedKFold(5, shuffle=True,
random_state=42 + rep), rep = 0, 1): 10 pliegues pareados, cada uno con ~371.800 celdas de
entrenamiento y ~93.000 de prueba. Con --holdout, entrena con las 464.809 celdas y predice la
prueba reservada de 116.203.

Cada modelo usa la mejor combinación del benchmark corregido (balanceo y configuración modal
de sus pliegues externos). Los hiperparámetros se eligieron con 40.000 celdas y se mantienen
fijos aquí para todos los modelos por igual (ver capítulo 12). SVM sigue limitado a 6.000
celdas de entrenamiento, como en su benchmark. HistGradientBoosting usa la configuración más
frecuente de su búsqueda en los 30 pliegues de 40.000 celdas.

Uso: python scripts/e3c_probas_completo.py --rep 0 --pliegue 3 [--hilos 3]
     python scripts/e3c_probas_completo.py --holdout [--hilos 12]
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import _e3 as e3
import _utils3 as u3

SALIDA = e3.RES_E3 / "completo"


def configurar_hilos(h):
    u3.MODELOS["RF"]["build"] = lambda **kw: RandomForestClassifier(random_state=u3.RANDOM_STATE, n_jobs=h, **kw)
    u3.MODELOS["KNN"]["build"] = lambda **kw: KNeighborsClassifier(n_jobs=h, **kw)
    u3.MODELOS["XGB"]["build"] = lambda **kw: XGBClassifier(
        objective="multi:softprob", num_class=7, eval_metric="mlogloss", tree_method="hist",
        random_state=u3.RANDOM_STATE, n_jobs=h, **kw)


def params_hgb():
    """Configuración más frecuente de HistGradientBoosting en los 30 pliegues de 40.000 celdas."""
    import glob
    vistos = []
    for f in glob.glob(str(e3.RES_E3 / "meta" / "rep*_f?.json")):
        vistos.append(json.dumps(json.load(open(f, encoding="utf-8"))["meta_probas"]["params_hgb"], sort_keys=True))
    return json.loads(Counter(vistos).most_common(1)[0][0])


def correr(X_tr, y_tr, X_te, y_te, destino, oof=True):
    mejores = e3.mejores_combinaciones()
    M, C = len(mejores), len(e3.CLASES)
    P_oof = np.zeros((M, len(X_tr), C), dtype=np.float32) if oof else None
    P_te = np.zeros((M, len(X_te), C), dtype=np.float32)
    tiempos, params = {}, {}
    for i, fila in mejores.iterrows():
        m, b, p = fila["modelo"], fila["balanceo"], fila["params_moda"]
        params[m] = p
        t0 = time.perf_counter()
        if oof:
            P_oof[i] = e3.probas_oof(m, b, p, X_tr, y_tr)
        t1 = time.perf_counter()
        pipe = e3.ajustar_modelo(m, b, p, X_tr, y_tr)
        t2 = time.perf_counter()
        P_te[i] = e3.proba_ordenada(pipe, m, X_te)
        t3 = time.perf_counter()
        tiempos[m] = {"oof_s": t1 - t0, "ajuste_s": t2 - t1, "inferencia_s": t3 - t2}
        print(f"{destino.name} {m}: F1={f1_score(y_te, e3.CLASES[P_te[i].argmax(1)], average='macro'):.4f} "
              f"({t3 - t0:.0f}s)", flush=True)
    ph = params_hgb()
    t0 = time.perf_counter()
    hgb = HistGradientBoostingClassifier(random_state=u3.RANDOM_STATE, early_stopping=False, **ph).fit(X_tr, y_tr)
    P_hgb = hgb.predict_proba(X_te).astype(np.float32)
    tiempos["HGB"] = {"ajuste_s": time.perf_counter() - t0}
    print(f"{destino.name} HGB: F1={f1_score(y_te, hgb.predict(X_te), average='macro'):.4f}", flush=True)
    extra = {"P_oof": P_oof} if oof else {}
    np.savez_compressed(destino, y_tr=y_tr.to_numpy(), y_te=y_te.to_numpy(), P_te=P_te, P_hgb=P_hgb,
                        modelos=np.array(mejores["modelo"].tolist()),
                        meta=json.dumps({"tiempos": tiempos, "params": params, "params_hgb": ph}), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int)
    ap.add_argument("--pliegue", type=int)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--hilos", type=int, default=3)
    a = ap.parse_args()
    configurar_hilos(a.hilos)
    SALIDA.mkdir(parents=True, exist_ok=True)
    _, _, X_test, y_test, X_train, y_train = u3.get_data(n=u3.N_BENCHMARK)
    if a.holdout:
        destino = SALIDA / "holdout.npz"
        if not destino.exists():
            correr(X_train, y_train, X_test, y_test, destino)
        return
    destino = SALIDA / f"rep{a.rep}_f{a.pliegue}.npz"
    if destino.exists():
        return
    cv = StratifiedKFold(5, shuffle=True, random_state=u3.RANDOM_STATE + a.rep)
    tr, te = list(cv.split(X_train, y_train))[a.pliegue]
    correr(X_train.iloc[tr], y_train.iloc[tr], X_train.iloc[te], y_train.iloc[te], destino)
    np.savez(SALIDA / f"rep{a.rep}_f{a.pliegue}_idx.npz", tr=tr, te=te)


if __name__ == "__main__":
    main()
