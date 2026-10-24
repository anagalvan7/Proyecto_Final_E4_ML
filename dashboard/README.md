# Dashboard · Clasificación de la cobertura forestal (Covertype) — Entregable 3

Aplicación Dash multipágina con tres secciones:

- **Contexto** (`/`): problema, pregunta de investigación, datos y diccionario de variables, pipeline
  metodológico, territorio, limitaciones y ética.
- **EDA** (`/eda`): filtros globales (área silvestre, cobertura y rango de elevación) que actualizan
  todos los gráficos; variable objetivo, η², distribuciones, correlación, explorador bivariado, faltantes y
  atípicos, y generalización a áreas no vistas.
- **Modelos** (`/modelos`): tabla de líderes (mejor combinación de cada uno de los 7 modelos, el modelo
  original ECLC y HistGradientBoosting), predicción en vivo con explicación local (KernelSHAP agrupado),
  diagnóstico por modelo (matriz de confusión, ROC, precisión-recall, calibración y SHAP global),
  comparación estadística (diagrama de diferencia crítica y pruebas pareadas con Holm) y la subsección del
  modelo original (pseudocódigo, diferencias con la literatura, convergencia, ablación, conjuntos adicionales
  y veredicto).

El dashboard **no entrena ni optimiza modelos**: carga una sola vez los artefactos de `artifacts/`.

## Ejecutar en local

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:8050
```

En Windows, si `xgboost-cpu` no está disponible para la plataforma, instala `xgboost==2.1.4`.

## Estructura

```
app.py               punto de entrada: crea la app (use_pages=True) y expone "server"
pages/               contexto.py · eda.py · modelos.py (una página por sección, con sus callbacks)
src/
  original_model.py  clase EnsambleCompetenciaLocal (modelo original, numpy)
  explicacion.py     KernelSHAP agrupado (numpy)
  modelos.py         carga de artefactos, validación de entradas y predicción
  data.py            carga y caché de datos, constantes y plantilla gráfica
  plots.py           figuras reutilizables
  ui.py              componentes visuales comunes
  seccion_e2.py      secciones del Entregable 2 que siguen vigentes
  preprocesamiento.py ingeniería de variables (la que referencian los artefactos)
artifacts/           modelos serializados (joblib, compress=3) y su metadata.json
datos/               datos livianos que usa la app (≈ 4 MB)
assets/              CSS, Bootstrap e ilustraciones
```

`artifacts/` y `datos/e3/` se generan desde la raíz del proyecto con los scripts `e3_*.py` (orden en el
capítulo 15 del Jupyter Book).

## Despliegue en Render

1. Subir el contenido de esta carpeta a un repositorio de GitHub (o el proyecto completo y fijar
   *Root Directory* = `dashboard`).
2. En Render: **New → Web Service** y conectar el repositorio (o **New → Blueprint**, que lee `render.yaml`).
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 120`
5. Abrir la URL unos minutos antes de la sustentación: el plan gratuito suspende el servicio tras un periodo
   de inactividad.

**Memoria.** Con los 8 modelos cargados, el proceso ocupó unos 500 MB en Windows. Para caber en los 512 MB
del plan gratuito se usa `xgboost-cpu` y un Random Forest de 80 árboles (los primeros 80 de los 200 del modelo
evaluado; el F1 macro en la prueba reservada cambia de 0,7934 a 0,7932). Si Render reporta *out of memory*,
revisar los registros y, como alternativa, usar el plan Starter.
