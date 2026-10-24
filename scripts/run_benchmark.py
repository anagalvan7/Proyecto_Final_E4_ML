"""Benchmark combinatorio de 7 modelos x 4 balanceo x 4 optimizacion (112
combinaciones), con validacion cruzada anidada.

Bucle externo (outer, 3 pliegues estratificados): estima el desempeno de generalizacion
del modelo ya optimizado, nunca se usa para elegir hiperparametros.
Bucle interno (dentro de cada entrenamiento del bucle externo): corre el metodo de
optimizacion completo (Grid, Random, Bayesiano o Genetico) con su propia validacion
cruzada de 3 pliegues, y devuelve los mejores hiperparametros encontrados.

Guarda resultados de forma incremental (un JSONL por combinacion terminada) para poder
reanudar si el proceso se interrumpe.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "benchmark_log.jsonl"  # se sobreescribe en main() si hay --modelos


def ya_hecho(clave_modelo, balanceo, opt, log_path=None):
    log_path = log_path or LOG_PATH
    if not log_path.exists():
        return False
    clave = f"{clave_modelo}|{balanceo}|{opt}"
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("clave") == clave and row.get("status") == "ok":
                return True
    return False


def guardar(resultado, log_path=None):
    log_path = log_path or LOG_PATH
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")


def correr_combinacion(clave_modelo, balanceo, opt, X, y, outer_cv, inner_cv):
    t_total0 = time.time()
    fold_scores_f1, fold_scores_acc, fold_params, fold_iters, fold_times = [], [], [], [], []
    fold_remuestreo_omitido = []
    n_evals_total = 0
    diversidad = None
    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        t0 = time.time()
        if opt == "genetico":
            mejor, historia, n_evals, div = u3.genetic_search(
                clave_modelo, balanceo, X_tr, y_tr, inner_cv, pop_size=4, n_gen=2
            )
            diversidad = div
        else:
            mejor, historia, n_evals = u3.OPTIMIZADORES[opt](clave_modelo, balanceo, X_tr, y_tr, inner_cv)
        t_inner = time.time() - t0

        if mejor is None:
            # combinacion no aplicable (ej. KNN + balanced): se documenta, no se cuenta como error
            return {"aplicable": False}

        y_tr_fit = (y_tr - 1) if clave_modelo == "XGB" else y_tr
        fit_params = u3._fit_params_for(clave_modelo, balanceo, y_tr_fit)
        pipe, remuestreo_omitido = u3.robust_fit(
            clave_modelo, balanceo, mejor["params"], X_tr, y_tr_fit, fit_params
        )
        pred = pipe.predict(X_te)
        if clave_modelo == "XGB":
            pred = pred + 1
        f1 = f1_score(y_te, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_te, pred)

        fold_scores_f1.append(f1)
        fold_scores_acc.append(acc)
        fold_params.append(mejor["params"])
        fold_iters.append(n_evals)
        fold_times.append(t_inner)
        fold_remuestreo_omitido.append(remuestreo_omitido)
        n_evals_total += n_evals

    return {
        "aplicable": True,
        "f1_macro_outer": fold_scores_f1,
        "accuracy_outer": fold_scores_acc,
        "mejores_hiperparametros_por_pliegue": fold_params,
        "evaluaciones_por_pliegue": fold_iters,
        "tiempo_por_pliegue_s": fold_times,
        "remuestreo_omitido_por_pliegue": fold_remuestreo_omitido,
        "evaluaciones_totales": n_evals_total,
        "tiempo_total_s": time.time() - t_total0,
        "diversidad_genetica": diversidad,
    }


def correr_una(clave):
    """Corre una combinacion con los mismos pliegues, semilla y submuestra del benchmark."""
    m, b, o = clave.split("|")
    out_dir = RESULTS_DIR / "repeticion"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"{m}_{b}_{o}.jsonl"
    if ya_hecho(m, b, o, log_path):
        print(f"{clave}: ya hecho", flush=True)
        return
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)
    X, y, _, _, _, _ = u3.get_data(n=u3.N_SVM if m == "SVM" else u3.N_BENCHMARK)
    t0 = time.time()
    r = correr_combinacion(m, b, o, X, y, outer_cv, inner_cv)
    r.update({"clave": clave, "modelo": m, "balanceo": b, "optimizacion": o, "status": "ok",
              "protocolo": "uniforme_E3"})
    guardar(r, log_path)
    f1m = np.mean(r["f1_macro_outer"]) if r.get("aplicable", True) else float("nan")
    print(f"{clave}: F1 macro={f1m:.4f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", default=None,
                        help="Lista separada por comas (ej. KNN,NB) para correr solo esos modelos "
                             "en este proceso; permite paralelizar entre modelos (Seccion 4.3).")
    parser.add_argument("--combo", default=None,
                        help="Una sola combinacion MODELO|balanceo|opt. Se registra en "
                             "resultados/repeticion/<MODELO>_<balanceo>_<opt>.jsonl para poder "
                             "correr muchas combinaciones en paralelo sin escribir al mismo archivo.")
    args = parser.parse_args()
    if args.combo:
        correr_una(args.combo)
        return
    modelos_filtro = args.modelos.split(",") if args.modelos else list(u3.MODELOS)

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)

    necesita_svm = "SVM" in modelos_filtro
    necesita_std = any(m != "SVM" for m in modelos_filtro)
    X_std = y_std = X_svm = y_svm = None
    if necesita_std:
        print("Cargando datos (submuestra estandar)...", flush=True)
        X_std, y_std, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    if necesita_svm:
        print("Cargando datos (submuestra SVM)...", flush=True)
        X_svm, y_svm, _, _, _, _ = u3.get_data(n=u3.N_SVM)

    combinaciones = [
        (m, b, o)
        for m in modelos_filtro
        for b in u3.BALANCEO
        for o in u3.OPTIMIZACION
    ]
    print(f"Modelos en este proceso: {modelos_filtro} · combinaciones: {len(combinaciones)}", flush=True)

    # Un proceso por modelo escribe a su propio archivo, para evitar escrituras
    # concurrentes al mismo JSONL desde varios procesos en paralelo (Seccion 4.3).
    if len(modelos_filtro) == 1:
        log_path = RESULTS_DIR / f"benchmark_log_{modelos_filtro[0]}.jsonl"
    else:
        log_path = LOG_PATH

    for i, (m, b, o) in enumerate(combinaciones, 1):
        clave = f"{m}|{b}|{o}"
        if ya_hecho(m, b, o, log_path):
            print(f"[{i}/{len(combinaciones)}] {clave}: ya hecho, se omite", flush=True)
            continue
        X, y = (X_svm, y_svm) if m == "SVM" else (X_std, y_std)
        t0 = time.time()
        try:
            r = correr_combinacion(m, b, o, X, y, outer_cv, inner_cv)
            r.update({"clave": clave, "modelo": m, "balanceo": b, "optimizacion": o, "status": "ok"})
            if not r.get("aplicable", True):
                print(f"[{i}/{len(combinaciones)}] {clave}: NO APLICABLE (balanceo no soportado)", flush=True)
            else:
                f1m = np.mean(r["f1_macro_outer"])
                print(f"[{i}/{len(combinaciones)}] {clave}: F1 macro={f1m:.4f} ({time.time()-t0:.0f}s)", flush=True)
            guardar(r, log_path)
        except Exception as e:
            print(f"[{i}/{len(combinaciones)}] {clave}: ERROR {e}", flush=True)
            traceback.print_exc()
            guardar({"clave": clave, "modelo": m, "balanceo": b, "optimizacion": o,
                      "status": "error", "error": str(e)}, log_path)

    print("BENCHMARK COMPLETO", flush=True)


if __name__ == "__main__":
    main()
