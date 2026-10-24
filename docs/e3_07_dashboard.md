# 13. Dashboard y despliegue

El dashboard pone los resultados del proyecto a disposición de una persona no experta: explica el problema,
permite explorar los datos y usa los modelos para predecir, sin entrenar nada en el servidor.

**URL pública:** *(completar con la URL de Render una vez desplegado)*.

## 13.1 Arquitectura

Es una aplicación Dash multipágina (`use_pages=True`) con una página por sección. La interfaz, la lógica de
datos y los artefactos de modelo están separados:

```
dashboard/
|-- app.py                 # crea la app, la barra de navegación fija y expone "server" (gunicorn)
|-- pages/
|   |-- contexto.py        # sección 1: Contexto
|   |-- eda.py             # sección 2: EDA (filtros globales + callbacks)
|   `-- modelos.py         # sección 3: Modelos (líderes, predicción, diagnóstico, estadística, ECLC)
|-- src/
|   |-- original_model.py  # clase EnsambleCompetenciaLocal
|   |-- explicacion.py     # KernelSHAP agrupado (numpy)
|   |-- modelos.py         # carga de artefactos (lru_cache), validación de entradas y predicción
|   |-- data.py            # carga única de datos, constantes y plantilla gráfica
|   |-- plots.py, ui.py    # figuras y componentes reutilizables
|   `-- seccion_e2.py      # secciones del Entregable 2 que siguen vigentes
|-- artifacts/             # 8 modelos (joblib, compress=3) + metadata.json de cada uno
|-- datos/                 # ≈ 4 MB de datos livianos
|-- assets/                # CSS, Bootstrap e ilustraciones
|-- requirements.txt       # versiones exactas
`-- render.yaml            # configuración de despliegue
```

## 13.2 Contenido por sección

| Sección | Contenido | Pregunta del usuario que responde |
|---|---|---|
| Contexto | Problema y relevancia (cifra de Blackard y Dean, 1999), pregunta y objetivos, fuente, diccionario de variables, diagrama del pipeline, territorio y especies, limitaciones, ética, autoras, versión y fecha | ¿Qué se predice, con qué datos y con qué límites? |
| EDA | Filtros globales (áreas, coberturas, rango de elevación) que actualizan todos los gráficos; distribución de la clase, η² por variable, distribuciones por cobertura, correlación, explorador bivariado (x, y y color), faltantes y atípicos, y generalización a áreas no vistas. Cada gráfico tiene una frase de interpretación, varias calculadas con los datos filtrados | ¿Qué variables separan las coberturas y dónde se confunden? |
| Modelos | Tabla de líderes ordenable y filtrable; predicción en vivo con validación de rangos, selector de modelo, probabilidades, explicación SHAP local y pesos del ECLC; diagnóstico (matriz de confusión, ROC, precisión-recall, calibración y SHAP global); diagrama de diferencia crítica y pruebas pareadas; subsección del modelo original con pseudocódigo, diferencias con la literatura, convergencia, ablación, conjuntos adicionales y veredicto explícito | ¿Qué modelo usar, qué tan bueno es y por qué predice lo que predice? |
| Benchmark (Entregable 2) | Modelo base, mapa de calor de las 112 combinaciones, comparación de optimizadores y eficiencia, con los resultados corregidos | ¿Cómo se llegó a los mejores modelos? |

## 13.3 Decisiones de diseño e ingeniería

- **Diseño visual.** Tema de Bootstrap con la paleta del proyecto: verde bosque como color principal, siete
  colores fijos para las coberturas y uno por modelo. Tipografía Nunito, barra de navegación fija y una
  plantilla de Plotly común (`bosque`) con títulos y ejes con unidades.
- **Rendimiento.** Datos y modelos se cargan una sola vez: los datos al importar `src/data.py` y los modelos
  con `functools.lru_cache` la primera vez que se usan. Los filtros del EDA usan una función con caché, y los
  callbacks de predicción y de ejemplos usan `prevent_initial_call` y `dash.no_update`.
- **Robustez.** El formulario valida tipos y rangos con el esquema de `metadata.json`. Una entrada inválida
  produce un `dbc.Alert` con el detalle y no detiene la aplicación. Las selecciones vacías del EDA muestran un
  mensaje en lugar de un gráfico roto.
- **Explicación local sin dependencias pesadas.** La librería `shap` (con numba) no cabe en la memoria del plan
  gratuito. La explicación local usa la implementación propia de KernelSHAP agrupado (sección 11.9), con 256
  coaliciones y 10 celdas de fondo: unos 3 s para el ECLC y menos de 1 s para los demás modelos.
- **Accesibilidad.** La información no depende solo del color: las barras llevan su valor como texto, el modelo
  original se marca además con una estrella y el diagrama del pipeline tiene texto alternativo.
- **Pruebas.** `tests/test_dashboard.py` construye las tres páginas y dispara los callbacks a través del
  servidor (EDA con filtros normales y vacíos, diagnóstico de los 8 modelos, predicción válida e inválida).

## 13.4 Artefactos y memoria

Cada modelo se serializó con `joblib.dump(..., compress=3)` junto con un `metadata.json` que incluye:

- versiones de las librerías y semilla;
- hiperparámetros finales;
- esquema de entrada (nombres, tipos y rangos válidos);
- métricas del bucle externo y de la prueba reservada;
- resultado de la prueba de ida y vuelta: predicciones idénticas del modelo en memoria y del recargado sobre
  2.000 celdas fijas de la prueba.

Todos los artefactos pasaron esa prueba.

Con los 8 modelos cargados, el proceso ocupó unos 625 MB, por encima de los 512 MB del plan gratuito de Render.
Se tomaron tres medidas, documentadas en `scripts/e3_reducir_rf.py` y en el `metadata.json` de Random Forest:

1. **Reducción de árboles equivalente en desempeño.** El artefacto desplegado de Random Forest usa los primeros
   80 de sus 200 árboles; con eso, el F1 macro en la prueba reservada cambia de 0,7934 a 0,7932 y el archivo
   pasa de 44,7 a 18 MB. El bosque completo se conserva en `artifacts_completos/` y es el que se usó en todas
   las métricas del libro.
2. `xgboost-cpu`, la distribución oficial de XGBoost sin CUDA, en lugar de `xgboost`.
3. Se eliminaron del arranque los cálculos y datos del finalista anterior (por ejemplo, `scipy.stats` y los
   valores SHAP de XGBoost).

## 13.5 Guía de despliegue

1. Verificar en local: `pip install -r requirements.txt` y `python app.py`, que expone `app.server`.
2. Subir la carpeta `dashboard/` a un repositorio de GitHub (o el proyecto con *Root Directory* = `dashboard`).
3. En Render, crear un **Web Service** conectado al repositorio, con *build command*
   `pip install -r requirements.txt` y *start command* `gunicorn app:server --workers 1 --threads 2 --timeout 120`
   (`render.yaml` ya los trae, con `PYTHON_VERSION = 3.10.16`).
4. Revisar los registros de construcción y comprobar en la URL pública las tres páginas y una predicción.
5. Abrir la URL unos minutos antes de la sustentación, porque el plan gratuito suspende el servicio inactivo.

```{admonition} Capturas de pantalla
:class: note
Agregar en esta sección las capturas de la versión desplegada (Contexto, EDA con filtros, predicción en vivo
con su explicación y subsección del modelo original) una vez publicada la URL.
```
