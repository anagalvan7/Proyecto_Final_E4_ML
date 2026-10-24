# Clasificación de la cobertura forestal en el Bosque Nacional Roosevelt (Covertype) — Entregable 4 (final)

Proyecto final de Machine Learning: datos y EDA, benchmark de 7 modelos × 4 estrategias de balanceo × 4
métodos de optimización con validación cruzada anidada (corregido para que las 112 combinaciones usen el mismo
espacio y presupuesto), un **modelo original** (ensamble por competencia local y por clase, ECLC) con su
validación estadística, y un dashboard en Dash desplegable en Render.

**Autoras:** Ana Galván, Valeria Oñate, María José Ruiz. **Semilla global:** 42.

- **Jupyter Book publicado:** https://anagalvan7.github.io/Proyecto_Final_E4_ML/
- **Dashboard (Render):** https://dashboard-covertype.onrender.com (plan gratuito: si estuvo inactivo, tarda
  cerca de un minuto en despertar)

## Estructura

```
datos/                    covertype.csv (UCI Covertype, CC BY 4.0)
Entregable4_Covertype.ipynb  todos los capítulos del libro en un solo cuaderno, con salidas
docs/                     Jupyter Book (fuentes .md MyST, _config.yml, _toc.yml, references.bib)
scripts/                  benchmark, corrección del protocolo y experimentos del Entregable 3 (e3_*.py)
resultados/               registros y tablas; entregable2_original/ guarda los registros sin corregir
resultados/e3/            pliegues, pruebas, literatura y tabla maestra del modelo original. Las probabilidades
                          intermedias por pliegue (*.npz, ~800 MB) no se versionan: se regeneran con scripts/e3*_probas*.py
dashboard/                aplicación Dash (app.py, pages/, src/, artifacts/, datos/, assets/, render.yaml)
artifacts_completos/      Random Forest de 200 árboles (el desplegado usa 80 por la memoria de Render)
tests/                    pruebas (pytest) del modelo original y del dashboard
```

El modelo original está en `dashboard/src/original_model.py` (clase `EnsambleCompetenciaLocal`).

## Instalación y uso

```bash
conda create -p ./env python=3.10 && conda activate ./env
pip install -r docs/requirements.txt pytest dash==2.18.2 dash-bootstrap-components==1.6.0 dash-iconify==0.1.2 plotly==5.24.1
python -m pytest tests                 # 32 pruebas
jupyter-book build docs                # libro en docs/_build/html/index.html
cd dashboard && python app.py          # dashboard en http://127.0.0.1:8050
```

El orden completo de ejecución de los experimentos está en el capítulo de reproducibilidad del libro.

## Despliegue en Render

En Render: **New → Blueprint**, conectar este repositorio y aplicar. `render.yaml` crea un servicio web
gratuito con raíz en `dashboard/` (`pip install -r requirements.txt`; `gunicorn app:server`). Más detalle en
`dashboard/README.md`.
