# Apéndice A. Reproducibilidad

## A.1 Estructura del proyecto

```text
proyecto-final-covertype/
├── datos/
│   └── covertype.csv                      Base de datos (581.012 filas, 55 columnas)
├── docs/                                  Jupyter Book
│   ├── _config.yml                        Configuración del libro
│   ├── _toc.yml                           Tabla de contenido
│   ├── _static/custom.css                 Estilos del libro
│   ├── requirements.txt                   Dependencias con versiones exactas
│   ├── references.bib                     Referencias bibliográficas
│   ├── _utils.py                          Carga, partición, ingeniería de variables y pipelines (Parte I)
│   ├── intro.md
│   ├── 01_base_datos.md … 10_conclusiones_finales.md
│   └── bibliografia.md
├── scripts/                               Código de la Parte II
│   ├── _utils3.py                         Algoritmos, espacios de búsqueda, balanceo y optimizadores
│   ├── run_benchmark.py                   Benchmark de 112 combinaciones con validación anidada
│   ├── analizar_resultados.py             Tabla maestra, Friedman y Nemenyi
│   ├── finalistas.py                      Reentrenamiento del finalista, DeLong y SHAP/LIME
│   ├── finalistas_extendido.py            Métricas, calibración, bootstrap, perfilamiento y semillas
│   ├── multifidelidad.py                  Successive Halving
│   └── experimentos_complementarios.py    Desempeño en cualquier momento, implementaciones
│                                          optimizadas, escalamiento, SHAP y áreas no vistas
└── resultados/                            Registros y tablas generadas por los scripts
```

La base de datos también puede descargarse del repositorio UCI
(<https://archive.ics.uci.edu/dataset/31/covertype>). El archivo local `covertype.csv`
corresponde al archivo original `covtype.data` con los nombres de las columnas agregados.

## A.2 Orden de ejecución

1. `run_benchmark.py --modelos <MODELO>` para cada uno de los siete algoritmos (ejecutados en
   procesos paralelos; sección 9.3). Cada proceso escribe
   `resultados/benchmark_log_<MODELO>.jsonl`.
2. `analizar_resultados.py`: consolida la tabla maestra y calcula Friedman y Nemenyi.
3. `finalistas.py` y `finalistas_extendido.py`: reentrenamiento y evaluación del finalista.
4. `multifidelidad.py` y `experimentos_complementarios.py anytime computacional
   escalamiento shap areas`.
5. `jupyter-book build docs`: compila el libro. Los capítulos de la Parte II solo leen los
   archivos de `resultados/`, de modo que la compilación no repite los
   experimentos.

## A.3 Entorno y semillas

- **Python:** 3.10.21, en un ambiente de conda dedicado.
- **Dependencias:** versiones exactas en `requirements.txt` (entre otras, scikit-learn
  1.7.2, imbalanced-learn 0.14.2, XGBoost 2.1.4, Optuna 5.0.0, DEAP 1.4.4, SHAP 0.49.1,
  LIME 0.2.0.1, scikit-posthocs 0.17.1, pandas 2.1.0, NumPy 1.25.2 y jupyter-book
  0.15.1).
- **Semilla aleatoria:** `RANDOM_STATE = 42`, definida en `_utils.py` y reexportada por
  `_utils3.py`. Se propaga a la partición entrenamiento/prueba, a todas las submuestras
  estratificadas, a los pliegues de validación cruzada, a SMOTE y ADASYN, a los modelos de
  scikit-learn y XGBoost, al muestreador TPE de Optuna, al generador aleatorio del
  algoritmo genético, al *bootstrap*, a Isolation Forest, a *k-means* y a t-SNE. Los
  análisis de estabilidad usan las semillas 0 a 4 de forma explícita.

## A.4 Créditos

La fotografía de la barra lateral, un bosque subalpino de coníferas en las Montañas
Rocosas de Colorado, proviene de *Jaye Travel Blog* (2022) y se usa con fines
exclusivamente académicos.

## A.5 Código auxiliar

El archivo `_utils.py` centraliza la carga del conjunto de datos, la partición
reproducible, la ingeniería de variables y la construcción de los *pipelines*, de modo
que todos los capítulos usan exactamente la misma partición y el mismo preprocesamiento.

```{literalinclude} _utils.py
:language: python
```
