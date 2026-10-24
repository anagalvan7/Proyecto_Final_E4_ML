"""Robustez, sensibilidad y complejidad empirica del ECLC (capitulo 5 del libro).

Sobre los 3 pliegues externos de la repeticion 0 (la particion del benchmark), usando las
probabilidades ya calculadas por e3_probas_pliegue.py:
  1. Sensibilidad a cada hiperparametro (k, beta, tau) variando uno y fijando los otros
     en los valores elegidos por la optimizacion del pliegue.
  2. Estabilidad ante la semilla del optimizador (TPE) y de la validacion interna: 5 semillas.
  3. Ruido en las etiquetas del conjunto de referencia (5 %, 10 %, 20 % de etiquetas cambiadas).
  4. Menor tamano de muestra: conjunto de referencia reducido al 10 %, 25 % y 50 %.
  5. Analisis de errores: F1 por clase y por area silvestre frente al mejor modelo base.
  6. Tiempo empirico de la capa de combinacion al variar n (referencia) y p (dimension).
Salida: resultados/e3/robustez.csv, robustez_errores.csv, complejidad_meta.csv
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import _e3 as e3
import _utils3 as u3
from e3_meta import eclc, f1m, optimizar
from src.original_model import EnsambleCompetenciaLocal

RNG = np.random.default_rng(42)


def cargar(f):
    d = np.load(e3.RES_E3 / "probas" / f"rep0_f{f}.npz")
    hp = json.load(open(e3.RES_E3 / "meta" / f"rep0_f{f}.json", encoding="utf-8"))["hiperparametros"]
    return d, hp


def evaluar(hp, X_tr, y_tr, P_oof, X_te, y_te, P_te):
    m = eclc(**hp).ajustar_desde_oof(X_tr, y_tr, P_oof)
    return f1m(y_te, m.combinar(P_te, X=X_te))


def main():
    X, _, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    filas, errores = [], []
    for f in range(3):
        d, hp = cargar(f)
        X_tr, X_te = X.iloc[d["idx_tr"]], X.iloc[d["idx_te"]]
        y_tr, y_te = d["y_tr"], d["y_te"]
        P_oof, P_te = d["P_oof"].astype(float), d["P_te"].astype(float)
        base = evaluar(hp, X_tr, y_tr, P_oof, X_te, y_te, P_te)
        filas.append({"experimento": "referencia", "pliegue": f, "valor": "elegido", "f1_macro": base})
        # 1. sensibilidad
        for k in [3, 6, 12, 25, 50, 100, 200]:
            filas.append({"experimento": "sensibilidad k", "pliegue": f, "valor": k,
                          "f1_macro": evaluar({**hp, "k": k}, X_tr, y_tr, P_oof, X_te, y_te, P_te)})
        for b in [0.0, 0.5, 2, 5, 10, 20, 50]:
            filas.append({"experimento": "sensibilidad beta", "pliegue": f, "valor": b,
                          "f1_macro": evaluar({**hp, "beta": b}, X_tr, y_tr, P_oof, X_te, y_te, P_te)})
        for t in [0.0, 1, 10, 100, 1000, 1e4, np.inf]:
            filas.append({"experimento": "sensibilidad tau", "pliegue": f, "valor": t,
                          "f1_macro": evaluar({**hp, "tau": t}, X_tr, y_tr, P_oof, X_te, y_te, P_te)})
        # 2. semillas del optimizador
        for s in range(5):
            hp_s, _ = optimizar(X_tr, y_tr, P_oof, semilla=100 + s)
            filas.append({"experimento": "semilla del optimizador", "pliegue": f, "valor": 100 + s,
                          "f1_macro": evaluar(hp_s, X_tr, y_tr, P_oof, X_te, y_te, P_te),
                          "hiperparametros": json.dumps(hp_s)})
        # 3. ruido en las etiquetas de referencia
        for tasa in [0.05, 0.10, 0.20]:
            y_r = y_tr.copy()
            idx = RNG.choice(len(y_r), int(tasa * len(y_r)), replace=False)
            y_r[idx] = RNG.choice(e3.CLASES, len(idx))
            filas.append({"experimento": "ruido en etiquetas", "pliegue": f, "valor": tasa,
                          "f1_macro": evaluar(hp, X_tr, y_r, P_oof, X_te, y_te, P_te)})
        # 4. tamano del conjunto de referencia
        for frac in [0.10, 0.25, 0.50]:
            sub, _ = u3.stratified_sample(pd.Series(np.arange(len(y_tr))), pd.Series(y_tr), int(frac * len(y_tr)))
            sub = sub.to_numpy()
            filas.append({"experimento": "tamano de referencia", "pliegue": f, "valor": frac,
                          "f1_macro": evaluar(hp, X_tr.iloc[sub], y_tr[sub], P_oof[:, sub], X_te, y_te, P_te)})
        # 5. errores por clase y area
        m = eclc(**hp).ajustar_desde_oof(X_tr, y_tr, P_oof)
        pred = e3.CLASES[m.combinar(P_te, X=X_te).argmax(1)]
        modelos = [str(x) for x in d["modelos"]]
        i_mejor = modelos.index("RF")
        pred_rf = e3.CLASES[P_te[i_mejor].argmax(1)]
        area = X_te[[f"Wilderness_Area_{i}" for i in range(1, 5)]].to_numpy().argmax(1) + 1
        for nombre, mask in [("todas", np.ones(len(y_te), bool))] + [(f"área {a}", area == a) for a in range(1, 5)]:
            for c, (fe, fr) in enumerate(zip(
                    f1_score(y_te[mask], pred[mask], labels=e3.CLASES, average=None, zero_division=np.nan),
                    f1_score(y_te[mask], pred_rf[mask], labels=e3.CLASES, average=None, zero_division=np.nan)), 1):
                errores.append({"pliegue": f, "area": nombre, "clase": c, "n": int(((y_te == c) & mask).sum()),
                                "f1_eclc": fe, "f1_rf": fr})
        print(f"pliegue {f} listo", flush=True)
    pd.DataFrame(filas).to_csv(e3.RES_E3 / "robustez.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errores).to_csv(e3.RES_E3 / "robustez_errores.csv", index=False, encoding="utf-8-sig")

    # 6. complejidad empirica de la capa de combinacion (datos sinteticos, M = 7, C = 7)
    tiempos = []
    for n in [2_500, 5_000, 10_000, 20_000, 40_000]:
        for p in [16, 57]:
            Z = RNG.normal(size=(n, p))
            y = RNG.integers(1, 8, n)
            P = RNG.dirichlet(np.ones(7), size=(7, n))
            Zq = RNG.normal(size=(2_000, p))
            Pq = RNG.dirichlet(np.ones(7), size=(7, 2_000))
            m = EnsambleCompetenciaLocal([("b", None)], k=25, beta=10, tau=10)
            t0 = time.perf_counter()
            m.ajustar_desde_oof(Z, y, P)
            t1 = time.perf_counter()
            m.combinar(Pq, X=Zq)
            t2 = time.perf_counter()
            tiempos.append({"n_referencia": n, "p": p, "ajuste_s": t1 - t0, "inferencia_2000_s": t2 - t1})
    pd.DataFrame(tiempos).to_csv(e3.RES_E3 / "complejidad_meta.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(filas).groupby(["experimento", "valor"])["f1_macro"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
