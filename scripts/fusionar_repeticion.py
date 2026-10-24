"""Agrega las combinaciones repetidas (resultados/repeticion/*.jsonl) a los registros
benchmark_log_<MODELO>.jsonl y regenera la tabla maestra y Friedman-Nemenyi.

Se ejecuta despues de scripts/uniformizar_protocolo.py y scripts/lanzar_repeticion.ps1.
"""
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"


def main():
    pendientes = [l.strip() for l in open(RES / "combinaciones_a_repetir.txt", encoding="utf-8") if l.strip()]
    faltan = []
    for clave in pendientes:
        m, b, o = clave.split("|")
        archivo = RES / "repeticion" / f"{m}_{b}_{o}.jsonl"
        filas = [json.loads(l) for l in open(archivo, encoding="utf-8")] if archivo.exists() else []
        filas = [r for r in filas if r.get("status") == "ok"]
        if not filas:
            faltan.append(clave)
            continue
        log = RES / f"benchmark_log_{m}.jsonl"
        ya = {json.loads(l)["clave"] for l in open(log, encoding="utf-8") if l.strip()}
        if clave not in ya:
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(filas[-1], ensure_ascii=False) + "\n")
    if faltan:
        sys.exit(f"Aun faltan {len(faltan)} combinaciones: {faltan}")
    subprocess.run([sys.executable, str(RAIZ / "scripts" / "analizar_resultados.py")], check=True)


if __name__ == "__main__":
    main()
