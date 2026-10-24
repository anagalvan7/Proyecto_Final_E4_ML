"""Agrega los resultados del benchmark (resultados/benchmark_log.jsonl) en:

- tabla_maestra.csv: una fila por combinación (modelo x balanceo x optimización), con
  F1 macro y accuracy (media +/- sd entre pliegues del bucle externo), hiperparámetros,
  tiempo y número de evaluaciones.
- friedman_nemenyi.csv / .json: prueba ómnibus de Friedman sobre las 112 combinaciones
  (usando los 3 F1 macro del bucle externo como "pliegues") y post-hoc de Nemenyi.
- No se ejecuta ningún modelo: solo lee lo que ya guardó run_benchmark.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _utils3 as u3  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "resultados"


def cargar_log():
    """Junta benchmark_log.jsonl y los benchmark_log_<MODELO>.jsonl de cada proceso paralelo."""
    filas = []
    for log_path in sorted(RESULTS_DIR.glob("benchmark_log*.jsonl")):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    filas.append(json.loads(line))
    return filas


def construir_tabla_maestra(filas):
    out = []
    for r in filas:
        if r.get("status") != "ok":
            continue
        fila = {
            "modelo": r["modelo"], "balanceo": r["balanceo"], "optimizacion": r["optimizacion"],
            "modelo_nombre": u3.NOMBRES_MODELOS.get(r["modelo"], r["modelo"]),
            "balanceo_nombre": u3.NOMBRES_BALANCEO.get(r["balanceo"], r["balanceo"]),
            "optimizacion_nombre": u3.NOMBRES_OPT.get(r["optimizacion"], r["optimizacion"]),
            "aplicable": r.get("aplicable", True),
        }
        if not r.get("aplicable", True):
            out.append(fila)
            continue
        f1s = r["f1_macro_outer"]
        accs = r["accuracy_outer"]
        fila.update({
            "f1_macro_media": float(np.mean(f1s)),
            "f1_macro_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            # json.dumps explicito: el str() por defecto de pandas usaria comillas simples
            # (repr de Python), que no son JSON valido y json.loads fallaria al releerlas.
            "f1_macro_pliegues": json.dumps(f1s),
            "accuracy_media": float(np.mean(accs)),
            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "tiempo_total_s": r.get("tiempo_total_s"),
            "evaluaciones_totales": r.get("evaluaciones_totales"),
            "mejores_hiperparametros_por_pliegue": json.dumps(r.get("mejores_hiperparametros_por_pliegue")),
            "remuestreo_omitido_algun_pliegue": any(r.get("remuestreo_omitido_por_pliegue", [False])),
        })
        out.append(fila)
    return pd.DataFrame(out)


def friedman_nemenyi(tabla):
    """Test de Friedman (ranking de cada combinación por pliegue) + post-hoc de Nemenyi."""
    aplicables = tabla[tabla["aplicable"]].copy()
    aplicables["combinacion"] = (
        aplicables["modelo_nombre"] + " · " + aplicables["balanceo_nombre"] + " · " + aplicables["optimizacion_nombre"]
    )
    pliegues_parseados = aplicables["f1_macro_pliegues"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else s
    )
    matriz = np.array(pliegues_parseados.tolist())  # filas=combinaciones, cols=pliegues
    n_combis, n_pliegues = matriz.shape
    # friedmanchisquare espera una muestra POR TRATAMIENTO (aquí, por combinación), cada una
    # con un valor por bloque (aquí, por pliegue del bucle externo): matriz[i, :], no matriz[:, j].
    stat, p = sstats.friedmanchisquare(*[matriz[i, :] for i in range(n_combis)])

    rangos = sstats.rankdata(-matriz, axis=0)  # rango 1 = mejor (F1 más alto) en cada pliegue
    rango_promedio = rangos.mean(axis=1)
    aplicables["rango_promedio"] = rango_promedio

    try:
        import scikit_posthocs as sp
        datos_largo = pd.DataFrame(matriz.T, columns=aplicables["combinacion"].values)
        datos_largo = datos_largo.melt(var_name="combinacion", value_name="f1")
        datos_largo["pliegue"] = np.tile(np.arange(n_pliegues), n_combis)
        nemenyi = sp.posthoc_nemenyi_friedman(matriz.T)
        nemenyi.index = aplicables["combinacion"].values
        nemenyi.columns = aplicables["combinacion"].values
    except Exception as e:
        nemenyi = None
        print("Nemenyi no se pudo calcular:", e)

    # Diferencia crítica (CD) de Nemenyi: q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha es el cuantil del rango estudentizado (Demsar 2006), calculado exactamente
    # con statsmodels en vez de aproximado, porque crece con k y una constante fija
    # subestima mucho la CD real para k grande (81-112 combinaciones).
    from statsmodels.stats.libqsturng import qsturng
    k = n_combis
    N = n_pliegues
    q_alpha = qsturng(0.95, k, 1e6) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    resumen = {
        "estadistico_friedman": float(stat), "p_valor": float(p), "grados_libertad": k - 1,
        "n_combinaciones": int(k), "n_pliegues": int(n_pliegues), "diferencia_critica_nemenyi": float(cd),
    }
    return aplicables.sort_values("rango_promedio"), nemenyi, resumen


def main():
    filas = cargar_log()
    print(f"Filas cargadas: {len(filas)}")
    tabla = construir_tabla_maestra(filas)
    tabla.to_csv(RESULTS_DIR / "tabla_maestra.csv", index=False, encoding="utf-8-sig")
    print(f"tabla_maestra.csv: {len(tabla)} filas ({tabla['aplicable'].sum()} aplicables)")

    if tabla["aplicable"].sum() >= 3:
        ranking, nemenyi, resumen = friedman_nemenyi(tabla)
        ranking.drop(columns=["f1_macro_pliegues", "mejores_hiperparametros_por_pliegue"]).to_csv(
            RESULTS_DIR / "friedman_ranking.csv", index=False, encoding="utf-8-sig"
        )
        if nemenyi is not None:
            nemenyi.to_csv(RESULTS_DIR / "nemenyi_pvalores.csv", encoding="utf-8-sig")
        with open(RESULTS_DIR / "friedman_resumen.json", "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        print("Friedman:", resumen)
    else:
        print("Muy pocas combinaciones aplicables todavía para Friedman; corre esto de nuevo al terminar el benchmark.")


if __name__ == "__main__":
    main()
