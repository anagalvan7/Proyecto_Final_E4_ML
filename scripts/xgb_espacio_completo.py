"""Re-ejecucion de XGBoost (Grid y Random Search) con el espacio de busqueda completo.

Elimina la asimetria descrita en la seccion 9.3: todas las estrategias de balanceo
(ninguno, SMOTE, ADASYN, pesos de clase) se optimizan sobre el mismo espacio
    n_estimators  {100, 200}
    max_depth     {4, 6, 8}
    learning_rate {0.05, 0.1, 0.2}
(18 configuraciones por pliegue con Grid; 8 configuraciones aleatorias con Random).
Mismo esquema de validacion anidada (3 x 3), misma submuestra y misma semilla que
run_benchmark.py.

Uso (una combinacion por proceso, para correrlas en paralelo):
    python scripts/xgb_espacio_completo.py --balanceo smote --opt grid
Cada combinacion se guarda en resultados/xgb_espacio_completo/<balanceo>_<opt>.jsonl.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402
import run_benchmark as rb  # noqa: E402

from sklearn.model_selection import StratifiedKFold  # noqa: E402
import numpy as np  # noqa: E402

ESPACIO_COMPLETO = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}
N_ITER_RANDOM = 8

SALIDA = rb.RESULTS_DIR / "xgb_espacio_completo"
SALIDA.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanceo", required=True, choices=u3.BALANCEO)
    parser.add_argument("--opt", required=True, choices=["grid", "random"])
    args = parser.parse_args()

    u3.MODELOS["XGB"]["espacio"] = ESPACIO_COMPLETO
    u3.OPTIMIZADORES["random"] = lambda cm, b, X, y, cv: u3.random_search(
        cm, b, X, y, cv, n_iter=N_ITER_RANDOM
    )

    log_path = SALIDA / f"{args.balanceo}_{args.opt}.jsonl"
    if rb.ya_hecho("XGB", args.balanceo, args.opt, log_path):
        print(f"XGB|{args.balanceo}|{args.opt}: ya hecho", flush=True)
        return

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u3.RANDOM_STATE)
    X, y, _, _, _, _ = u3.get_data(n=u3.N_BENCHMARK)

    clave = f"XGB|{args.balanceo}|{args.opt}"
    print(f"{clave}: inicio {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    r = rb.correr_combinacion("XGB", args.balanceo, args.opt, X, y, outer_cv, inner_cv)
    r.update({"clave": clave, "modelo": "XGB", "balanceo": args.balanceo,
              "optimizacion": args.opt, "status": "ok", "espacio": "completo"})
    rb.guardar(r, log_path)
    print(f"{clave}: F1 macro = {np.mean(r['f1_macro_outer']):.4f} "
          f"({(time.time() - t0) / 60:.0f} min) -> {log_path.name}", flush=True)


if __name__ == "__main__":
    main()
