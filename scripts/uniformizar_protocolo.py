"""Correccion del Entregable 2: protocolo de optimizacion identico para las 112 combinaciones.

Durante el Entregable 2 el espacio de busqueda de RF y XGBoost y los presupuestos de
Random Search, Bayesiano y Genetico se redujeron con el benchmark ya en marcha (seccion
9.3 del libro del Entregable 2). Quedaron dos grupos de corridas con condiciones distintas:

- Espacio grande: XGB y RF sin balanceo con Grid y Random Search.
- Presupuesto grande: Random Search con 8 iteraciones (en vez de 6), Bayesiano con 10
  ensayos (en vez de 6) y Genetico con 6 individuos x 3 generaciones (en vez de 4 x 2).

Este script:
1. Copia los registros originales a resultados/entregable2_original/ (no se borran).
2. Deja en resultados/benchmark_log_<M>.jsonl solo las corridas que ya cumplen el
   protocolo unico definido en _utils3.py (espacio reducido; 6 / 6 / 4x2).
3. Escribe resultados/combinaciones_a_repetir.txt con las claves que faltan.

Uso: python scripts/uniformizar_protocolo.py
"""
import json
import shutil
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "resultados"
ORIG = RES / "entregable2_original"

PRESUPUESTO = {"random": 6, "bayes": 6}
MAX_EVALS_GENETICO = 4 + 2 * 3  # poblacion 4 + 2 generaciones x 3 descendientes como maximo


def dentro_del_espacio(modelo, params):
    """True si cada hiperparametro elegido esta dentro del espacio reducido actual."""
    espacio = u3.MODELOS[modelo]["espacio"]
    tipos = u3.MODELOS[modelo]["tipo"]
    for k, v in params.items():
        if k not in espacio:
            continue
        if tipos[k] == "logfloat":
            lo, hi = min(espacio[k]), max(espacio[k])
            if not (lo - 1e-12 <= v <= hi + 1e-12):
                return False
        elif v not in espacio[k]:
            return False
    return True


def cumple_protocolo(row):
    if row.get("status") != "ok":
        return False
    if not row.get("aplicable", True):
        return True
    m, opt = row["modelo"], row["optimizacion"]
    evals = row["evaluaciones_por_pliegue"]
    if opt == "grid":
        tam = len(list(product(*u3.MODELOS[m]["espacio"].values())))
        ok_presupuesto = all(e == tam for e in evals)
    elif opt in PRESUPUESTO:
        ok_presupuesto = all(e == PRESUPUESTO[opt] for e in evals)
    else:
        ok_presupuesto = all(e <= MAX_EVALS_GENETICO for e in evals)
    ok_espacio = all(dentro_del_espacio(m, p) for p in row["mejores_hiperparametros_por_pliegue"])
    return ok_presupuesto and ok_espacio


def main():
    ORIG.mkdir(exist_ok=True)
    pendientes = []
    for m in u3.MODELOS:
        log = RES / f"benchmark_log_{m}.jsonl"
        respaldo = ORIG / log.name
        if not respaldo.exists():
            shutil.copy2(log, respaldo)
        filas = [json.loads(l) for l in open(respaldo, encoding="utf-8") if l.strip()]
        conservadas = [r for r in filas if cumple_protocolo(r)]
        claves_ok = {r["clave"] for r in conservadas}
        with open(log, "w", encoding="utf-8") as f:
            for r in conservadas:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for b in u3.BALANCEO:
            for o in u3.OPTIMIZACION:
                clave = f"{m}|{b}|{o}"
                if clave not in claves_ok:
                    pendientes.append(clave)
    (RES / "combinaciones_a_repetir.txt").write_text("\n".join(pendientes) + "\n", encoding="utf-8")
    print(f"Combinaciones a repetir: {len(pendientes)}")
    for c in pendientes:
        print("  ", c)


if __name__ == "__main__":
    main()
