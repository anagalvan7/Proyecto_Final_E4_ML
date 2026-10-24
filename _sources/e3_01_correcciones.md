---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Apéndice B. Registro de cambios del benchmark

Este apéndice documenta los ajustes hechos sobre una ejecución previa del benchmark y su efecto en los
resultados. Su objetivo es que las cifras de la Parte II puedan rastrearse hasta los registros originales.

```{code-cell} ipython3
:tags: [hide-input]

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path.cwd().parent
sys.path.insert(0, str(RAIZ / "scripts"))
import _utils3 as u3

RES = RAIZ / "resultados"


def cargar(carpeta):
    filas = {}
    for m in u3.MODELOS:
        for l in open(carpeta / f"benchmark_log_{m}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            filas[r["clave"]] = r
    return filas


antes = cargar(RES / "entregable2_original")
despues = cargar(RES)
repetidas = [l.strip() for l in open(RES / "combinaciones_a_repetir.txt", encoding="utf-8") if l.strip()]
```

## B.1 Cambios realizados

| # | Situación detectada | Acción | Ubicación | Evidencia |
|---|---|---|---|---|
| 1 | En una ejecución previa, XGBoost sin balanceo (Grid y Random Search) se optimizó con un espacio mayor ({100, 200} × {4, 6, 8} × {0,05; 0,1; 0,2}) que las demás variantes de XGBoost | Las 112 combinaciones se igualaron al mismo espacio por algoritmo y se repitieron las afectadas con la misma submuestra, pliegues y semilla | Secciones 5.2 y 9.3 | Tabla B.2 |
| 2 | Lo mismo en Random Forest sin balanceo ({100, 200, 300} × {10, 20, sin límite}) | Igual que en la fila 1 | Sección 5.2 | Tabla B.2 (filas RF) |
| 3 | Presupuestos distintos entre combinaciones (Random Search con 8 o 6 iteraciones, Bayesiana con 10 o 6 ensayos, Genético con 6 × 3 o 4 × 2) | Todas se repitieron con el presupuesto común (6, 6 y 4 × 2) | Secciones 5.2 y 6.2 | Tabla B.2 y conteos de B.3 |
| 4 | Resultados que dependían de la ejecución previa (ranking, finalista, comparación estadística) | Recalculados con el benchmark homogéneo | Capítulos 5 a 9 | `tabla_maestra.csv`, `friedman_resumen.json`, `resumen_finalista.json` |
| 5 | En la prueba de Friedman, los pliegues del SVM (submuestra de 9.000 celdas) se emparejaban con pliegues de otra submuestra (40.000) | Toda comparación del modelo original usa los mismos pliegues de la submuestra de 40.000 celdas para todos los modelos | Sección 12.1 | Resultados por pliegue |
| 6 | Año de publicación de Blackard y Dean | Corregido a 1999 | Bibliografía | `references.bib` |

## B.2 Combinaciones repetidas

```{code-cell} ipython3
:tags: [hide-input]

filas = []
for clave in repetidas:
    a, d = antes[clave], despues[clave]
    filas.append({
        "Combinación": clave.replace("|", " · "),
        "Evaluaciones por pliegue (previa)": a["evaluaciones_por_pliegue"][0],
        "Evaluaciones por pliegue (homogénea)": d["evaluaciones_por_pliegue"][0],
        "F1 macro previa": np.mean(a["f1_macro_outer"]),
        "F1 macro homogénea": np.mean(d["f1_macro_outer"]),
        "Cambio": np.mean(d["f1_macro_outer"]) - np.mean(a["f1_macro_outer"]),
    })
print(f"Combinaciones repetidas: {len(filas)} de 112")
pd.DataFrame(filas).round(4)
```

La tabla compara, para las combinaciones repetidas, el número de evaluaciones del optimizador por pliegue
externo y el F1 macro medio del bucle externo en la ejecución previa y en la homogénea. Todo lo demás se mantuvo
fijo. Muchas combinaciones de Random y Bayes no cambian, porque con la misma semilla sus primeras 6
configuraciones coinciden con las de la ejecución previa.

## B.3 Presupuesto por método

```{code-cell} ipython3
:tags: [hide-input]

def presupuesto(filas, metodo):
    return sorted({r["evaluaciones_por_pliegue"][0] for r in filas.values()
                   if r.get("aplicable", True) and r["optimizacion"] == metodo})

for metodo in ["grid", "random", "bayes", "genetico"]:
    print(f"{metodo:9s} evaluaciones por pliegue  previa: {presupuesto(antes, metodo)}   homogénea: {presupuesto(despues, metodo)}")
```

## B.4 Efecto en el ranking

```{code-cell} ipython3
:tags: [hide-input]

def mejores(filas):
    df = pd.DataFrame([{"modelo": r["modelo"], "balanceo": r["balanceo"], "optimizacion": r["optimizacion"],
                        "f1": np.mean(r["f1_macro_outer"])} for r in filas.values() if r.get("aplicable", True)])
    return df.loc[df.groupby("modelo")["f1"].idxmax()].set_index("modelo")

ma, md = mejores(antes), mejores(despues)
pd.DataFrame({"Mejor combinación (previa)": ma["balanceo"] + " · " + ma["optimizacion"], "F1 previa": ma["f1"],
              "Mejor combinación (homogénea)": md["balanceo"] + " · " + md["optimizacion"], "F1 homogénea": md["f1"],
              "Rango homogéneo": md["f1"].rank(ascending=False).astype(int)}).sort_values("Rango homogéneo").round(4)
```

Con el benchmark homogéneo, XGBoost pasa del primer lugar (0,809) al tercero (0,746). Su ventaja en la
ejecución previa se debía al espacio de búsqueda mayor que solo tuvieron sus variantes sin balanceo. Random
Forest con SMOTE pasa a ser la mejor combinación (0,783), y el balanceo mejora a Random Forest en lugar de no
afectarlo.
