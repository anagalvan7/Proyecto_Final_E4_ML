"""Optimizacion multi-fidelidad (Successive Halving) sobre Random Forest.

Fidelidad usada: n_estimators (numero de arboles). La intuicion es que el ranking
relativo de configuraciones con pocos arboles ya anticipa, en buena medida, el ranking
con muchos arboles, asi que no hace falta gastar el presupuesto completo en las
configuraciones que claramente van peor.

Factor de reduccion eta=2: en cada ronda se descarta la mitad de las configuraciones
peor rankeadas y se duplica el presupuesto (n_estimators) de las que sobreviven.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

RESULTS3 = Path(__file__).resolve().parents[1] / "resultados"

CANDIDATOS = [
    {"max_depth": 5, "min_samples_leaf": 1}, {"max_depth": 5, "min_samples_leaf": 10},
    {"max_depth": 10, "min_samples_leaf": 1}, {"max_depth": 10, "min_samples_leaf": 10},
    {"max_depth": 15, "min_samples_leaf": 1}, {"max_depth": 15, "min_samples_leaf": 10},
    {"max_depth": 20, "min_samples_leaf": 1}, {"max_depth": 20, "min_samples_leaf": 10},
]
RUNGS = [20, 40, 80, 160]  # n_estimators por ronda (eta=2: se duplica cada ronda)


def evaluar(config, n_estimators, X, y, cv):
    pipe = SkPipeline(list(u3.build_preprocessor_steps()) + [
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1, **config))
    ])
    scores, tiempos = [], []
    for tr, va in cv.split(X, y):
        t0 = time.time()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        tiempos.append(time.time() - t0)
        pred = pipe.predict(X.iloc[va])
        scores.append(f1_score(y.iloc[va], pred, average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.sum(tiempos))


def main():
    X, y, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    supervivientes = list(range(len(CANDIDATOS)))
    historial = []
    costo_acumulado_sh = 0.0

    for ronda, n_est in enumerate(RUNGS):
        print(f"Ronda {ronda} · presupuesto n_estimators={n_est} · {len(supervivientes)} configuraciones", flush=True)
        resultados_ronda = []
        for idx in supervivientes:
            score, tiempo = evaluar(CANDIDATOS[idx], n_est, X, y, cv)
            resultados_ronda.append({"candidato": idx, "config": CANDIDATOS[idx], "n_estimators": n_est,
                                      "f1_macro": score, "tiempo_s": tiempo})
            costo_acumulado_sh += tiempo
            print(f"  candidato {idx} {CANDIDATOS[idx]}: F1={score:.4f} ({tiempo:.1f}s)", flush=True)
        historial.extend(resultados_ronda)
        resultados_ronda.sort(key=lambda r: r["f1_macro"], reverse=True)
        n_sobreviven = max(1, len(supervivientes) // 2)
        supervivientes = [r["candidato"] for r in resultados_ronda[:n_sobreviven]]
        if len(supervivientes) == 1 and n_est == RUNGS[-1]:
            break

    ganador = supervivientes[0]
    print(f"Ganador: candidato {ganador} = {CANDIDATOS[ganador]}", flush=True)

    # Costo de comparacion: evaluar las 8 configuraciones completas a presupuesto maximo (sin SH)
    print("Calculando costo de referencia (grid search pleno, sin reduccion sucesiva)...", flush=True)
    costo_sin_sh = 0.0
    for config in CANDIDATOS:
        _, tiempo = evaluar(config, RUNGS[-1], X, y, cv)
        costo_sin_sh += tiempo

    resumen = {
        "ganador": CANDIDATOS[ganador], "costo_successive_halving_s": costo_acumulado_sh,
        "costo_grid_pleno_s": costo_sin_sh,
        "ahorro_relativo": 1 - costo_acumulado_sh / costo_sin_sh,
        "factor_reduccion_eta": 2, "presupuestos_n_estimators": RUNGS,
    }
    pd.DataFrame(historial).to_csv(RESULTS3 / "multifidelidad_historial.csv", index=False, encoding="utf-8-sig")
    with open(RESULTS3 / "multifidelidad_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(resumen)


if __name__ == "__main__":
    main()
