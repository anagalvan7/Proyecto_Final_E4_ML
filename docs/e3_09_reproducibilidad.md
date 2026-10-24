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

# 14. Reproducibilidad

## 14.1 Entorno y semillas

- **Python 3.10**, con las versiones exactas fijadas en `docs/requirements.txt`. Las principales son
  scikit-learn 1.7.2, imbalanced-learn 0.14.2, XGBoost 2.1.4, Optuna 5.0.0, numpy 1.25.2 y pandas 2.1.0.
- **Semilla global** `RANDOM_STATE = 42`. Se propaga a la partición 80/20, a la submuestra de 40.000 celdas, a
  los pliegues externos e internos, a los modelos, a SMOTE y ADASYN, a Optuna (TPE), al algoritmo genético y a
  las predicciones fuera de pliegue del modelo original. La repetición $r$ de la validación repetida usa la
  semilla $42 + r$.
- **Hardware.** Los experimentos se ejecutaron en dos estaciones de trabajo con Windows (12 y 24 hilos
  lógicos), con `n_jobs = 1` por estimador y procesos paralelos por combinación o por pliegue. Los tiempos
  absolutos de equipos distintos no son comparables (sección 6.2).

## 14.2 Orden de ejecución

Desde la raíz del proyecto, con el ambiente activado:

| Paso | Comando | Produce |
|---|---|---|
| 1 | `python scripts/run_benchmark.py --modelos M` (un proceso por algoritmo) | Registros de las 112 combinaciones |
| 2 | `python scripts/analizar_resultados.py` | `tabla_maestra.csv`, Friedman-Nemenyi |
| 3 | `python scripts/e3_probas_pliegue.py --rep r --pliegue f` (r = 0…9, f = 0…2) | Probabilidades de los 7 modelos en los 30 pliegues |
| 4 | `python scripts/e3_meta.py --rep r --pliegue f` | Modelo original, ablación y reglas de referencia por pliegue |
| 5 | `python scripts/e3_holdout_artefactos.py` | Modelos finales y evaluación en la prueba reservada |
| 6 | `python scripts/e3_estadistica.py` | Pruebas de hipótesis y diagrama de diferencia crítica |
| 7 | `python scripts/e3_robustez.py` | Sensibilidad, robustez y complejidad empírica |
| 8 | `python scripts/e3_datasets_adicionales.py --conjunto c` y luego sin argumentos | Conjuntos de datos adicionales |
| 9 | `python scripts/e3_finalista_completo.py` | Finalista del benchmark con el entrenamiento completo |
| 10 | `python scripts/e3c_probas_completo.py --rep r --pliegue f` (r = 0, 1; f = 0…4) y `--holdout` | Probabilidades con el entrenamiento completo |
| 11 | `python scripts/e3c_meta.py --rep r --pliegue f` y `--holdout` | Modelo original con el entrenamiento completo |
| 12 | `python scripts/e3c_estadistica.py` | Pruebas de hipótesis con el entrenamiento completo |
| 13 | `python scripts/e3_revision_literatura.py` | Registro de las búsquedas bibliográficas |
| 14 | `python -m pytest tests` | Pruebas unitarias del modelo original |
| 15 | `jupyter-book build docs` | Este documento |

Los pasos 3, 4, 8, 10 y 11 se ejecutan en paralelo, con un proceso por pliegue o por conjunto y
`OMP_NUM_THREADS = 1`.

## 14.3 Tabla maestra de experimentos

```{code-cell} ipython3
:tags: [hide-input]

from pathlib import Path
import pandas as pd

t = pd.read_csv(Path.cwd().parent / "resultados" / "e3" / "tabla_maestra_e3.csv", encoding="utf-8-sig")
print(f"{len(t)} filas: 112 combinaciones del benchmark + modelo original + HistGradientBoosting")
t[["modelo", "balanceo", "optimizacion", "aplicable", "f1_macro_media", "f1_macro_sd", "accuracy_media",
   "tiempo_total_s", "semilla", "particion"]].sort_values("f1_macro_media", ascending=False).head(15).round(4)
```

La tabla maestra (`resultados/e3/tabla_maestra_e3.csv`) contiene, por combinación, los hiperparámetros elegidos
en cada pliegue, las métricas del bucle externo, el número de evaluaciones, los tiempos, la semilla y la
partición. Las filas del modelo original y de HistGradientBoosting resumen los 30 pliegues de la validación
repetida.
