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

# 1. Base de datos

```{code-cell} ipython3
:tags: [hide-input]

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _utils import (
    BIN_COLS, CLASS_NAMES, CLASS_PALETTE, DISPLAY_NAMES, FEATURES, NUM_COLS, RANDOM_STATE,
    SOIL_COLS, TARGET, WILD_COLS, WILD_NAMES, load_covertype, split_covertype, wilderness_label,
)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 30)

df, X, y = load_covertype()
X_train, X_test, y_train, y_test = split_covertype(X, y)
```

## 1.1 Definición del problema

El objetivo es **clasificar el tipo de cobertura forestal dominante** de una celda de terreno de 30 × 30 m a partir de variables cartográficas. La variable objetivo `Cover_Type` toma siete valores, cada uno correspondiente a una especie o asociación forestal dominante. Las predictoras describen la topografía (elevación, orientación, pendiente y sombreado), la proximidad a elementos del paisaje (agua superficial, carreteras y puntos de ignición de incendios históricos), el área silvestre a la que pertenece la celda y su tipo de suelo.

Formalmente, se busca una función $f: \mathbb{R}^{54} \to \{1, \dots, 7\}$ que asigne a cada celda su clase de cobertura, aprendida a partir de ejemplos etiquetados y evaluada sobre celdas no vistas durante el entrenamiento.

## 1.2 Justificación de la selección

La elección del conjunto *Covertype* se apoya en cuatro razones:

1. **Relevancia aplicada.** La cobertura forestal condiciona la gestión de bosques, el riesgo y el comportamiento de incendios, la conservación de hábitats y la regulación hídrica. Inferirla a partir de cartografía reduce la dependencia de inventarios de campo costosos.
2. **Tamaño y calidad.** Con 581.012 observaciones, el conjunto es lo bastante grande para sostener validación cruzada anidada y particiones de prueba holgadas sin agotar las clases minoritarias; además no tiene valores faltantes y proviene de fuentes oficiales (US Forest Service y US Geological Survey).
3. **Dificultad adecuada.** Es un problema multiclase con desbalance fuerte y fronteras de decisión no lineales. Es además un conjunto de referencia habitual en la evaluación de algoritmos para datos tabulares {cite}`grinsztajn2022`. Un clasificador lineal obtiene un desempeño intermedio, lo que deja margen de mejora real para modelos más complejos y evita un problema trivial.
4. **Baja exposición a fuga de información.** Las variables son atributos físicos del terreno medidos de forma independiente de la etiqueta, y no existen identificadores ni variables derivadas de la cobertura.

## 1.3 Fuente de los datos y licencia

| Atributo | Detalle |
|---|---|
| Nombre | Covertype |
| Repositorio | UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/31/covertype> |
| DOI | [10.24432/C50K5N](https://doi.org/10.24432/C50K5N) |
| Autor original | Jock A. Blackard (US Forest Service), con Denis J. Dean y Charles W. Anderson (Colorado State University) |
| Fuentes primarias | Sistema de Información de Recursos de la Región 2 del US Forest Service (cobertura) y datos del US Geological Survey y del USFS (variables cartográficas) |
| Fecha de donación | 1 de agosto de 1998 |
| Licencia | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Referencia | {cite}`blackard2000` |

La licencia CC BY 4.0 permite usar, compartir y adaptar los datos con fines académicos siempre que se reconozca la autoría, como se hace en este documento.

## 1.4 Diccionario de variables

| Variable | Tipo | Unidad | Significado |
|---|---|---|---|
| `Elevation` | Numérica continua | m | Elevación sobre el nivel del mar |
| `Aspect` | Numérica circular | ° azimut (0–360) | Orientación de la ladera respecto al norte |
| `Slope` | Numérica continua | ° | Pendiente del terreno |
| `Horizontal_Distance_To_Hydrology` | Numérica continua | m | Distancia horizontal a la fuente de agua superficial más cercana |
| `Vertical_Distance_To_Hydrology` | Numérica continua | m | Diferencia de altura con esa fuente de agua (negativa si la celda está por debajo) |
| `Horizontal_Distance_To_Roadways` | Numérica continua | m | Distancia horizontal a la carretera más cercana |
| `Hillshade_9am` | Numérica discreta | índice 0–255 | Iluminación relativa a las 9 a. m. en el solsticio de verano |
| `Hillshade_Noon` | Numérica discreta | índice 0–255 | Iluminación relativa al mediodía en el solsticio de verano |
| `Hillshade_3pm` | Numérica discreta | índice 0–255 | Iluminación relativa a las 3 p. m. en el solsticio de verano |
| `Horizontal_Distance_To_Fire_Points` | Numérica continua | m | Distancia horizontal al punto de ignición de incendio forestal más cercano |
| `Wilderness_Area_1` … `_4` | Binaria (*one-hot*) | 0/1 | Área silvestre: 1 Rawah, 2 Neota, 3 Comanche Peak, 4 Cache la Poudre |
| `Soil_Type_1` … `_40` | Binaria (*one-hot*) | 0/1 | Tipo de suelo según la clasificación de unidades ecológicas terrestres (ELU) del USFS |
| `Cover_Type` | Categórica nominal (objetivo) | 1–7 | Tipo de cobertura forestal dominante |

Las siete clases de la variable objetivo son:

```{code-cell} ipython3
:tags: [hide-input]

pd.DataFrame(
    {"Clase": list(CLASS_NAMES.keys()), "Cobertura": list(CLASS_NAMES.values())}
).set_index("Clase")
```

Los 40 tipos de suelo corresponden a unidades ELU del US Forest Service. Su descripción completa se incluye a continuación de forma plegable, porque resulta útil para interpretar los coeficientes del modelo, pero no para la lectura general.

```{code-cell} ipython3
:tags: [hide-input, hide-output]

soil_desc = [
    (2702, "Cathedral family - Rock outcrop complex, extremely stony"),
    (2703, "Vanet - Ratake families complex, very stony"),
    (2704, "Haploborolis - Rock outcrop complex, rubbly"),
    (2705, "Ratake family - Rock outcrop complex, rubbly"),
    (2706, "Vanet family - Rock outcrop complex, rubbly"),
    (2717, "Vanet - Wetmore families - Rock outcrop complex, stony"),
    (3501, "Gothic family"),
    (3502, "Supervisor - Limber families complex"),
    (4201, "Troutville family, very stony"),
    (4703, "Bullwark - Catamount families - Rock outcrop complex, rubbly"),
    (4704, "Bullwark - Catamount families - Rock land complex, rubbly"),
    (4744, "Legault family - Rock land complex, stony"),
    (4758, "Catamount family - Rock land - Bullwark family complex, rubbly"),
    (5101, "Pachic Argiborolis - Aquolis complex"),
    (5151, "Sin especificar en el inventario de suelos del USFS"),
    (6101, "Cryaquolis - Cryoborolis complex"),
    (6102, "Gateview family - Cryaquolis complex"),
    (6731, "Rogert family, very stony"),
    (7101, "Typic Cryaquolis - Borohemists complex"),
    (7102, "Typic Cryaquepts - Typic Cryaquolls complex"),
    (7103, "Typic Cryaquolls - Leighcan family, till substratum complex"),
    (7201, "Leighcan family, till substratum, extremely bouldery"),
    (7202, "Leighcan family, till substratum - Typic Cryaquolls complex"),
    (7700, "Leighcan family, extremely stony"),
    (7701, "Leighcan family, warm, extremely stony"),
    (7702, "Granile - Catamount families complex, very stony"),
    (7709, "Leighcan family, warm - Rock outcrop complex, extremely stony"),
    (7710, "Leighcan family - Rock outcrop complex, extremely stony"),
    (7745, "Como - Legault families complex, extremely stony"),
    (7746, "Como family - Rock land - Legault family complex, extremely stony"),
    (7755, "Leighcan - Catamount families complex, extremely stony"),
    (7756, "Catamount family - Rock outcrop - Leighcan family complex, extremely stony"),
    (7757, "Leighcan - Catamount families - Rock outcrop complex, extremely stony"),
    (7790, "Cryorthents - Rock land complex, extremely stony"),
    (8703, "Cryumbrepts - Rock outcrop - Cryaquepts complex"),
    (8707, "Bross family - Rock land - Cryumbrepts complex, extremely stony"),
    (8708, "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony"),
    (8771, "Leighcan - Moran families - Cryaquolls complex, extremely stony"),
    (8772, "Moran family - Cryorthents - Leighcan family complex, extremely stony"),
    (8776, "Moran family - Cryorthents - Rock land complex, extremely stony"),
]
pd.DataFrame(
    [(f"Soil_Type_{i}", code, desc) for i, (code, desc) in enumerate(soil_desc, start=1)],
    columns=["Variable", "Código ELU", "Descripción"],
).set_index("Variable")
```

## 1.5 Estructura de los datos

```{code-cell} ipython3
:tags: [hide-input]

estructura = pd.DataFrame(
    {
        "Tipo de dato en el archivo": df.dtypes.astype(str).value_counts(),
    }
)
print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]} (54 predictoras + 1 objetivo)".replace(",", "."))
print(f"Predictoras numéricas: {len(NUM_COLS)} | Indicadoras de área silvestre: {len(WILD_COLS)} | Indicadoras de suelo: {len(SOIL_COLS)}")
estructura
```

- **Unidad de observación:** una celda de 30 × 30 m del terreno.
- **Nivel de agregación:** cada fila resume la celda completa; la cobertura corresponde al tipo dominante en ella según el sistema de información del USFS, y las variables cartográficas se derivan de modelos digitales de elevación y capas del USGS y del USFS.
- **Tipo de datos:** transversal. No hay variable temporal: la información describe el estado del bosque en el momento de construcción de la cartografía, y todas las celdas corresponden a una misma fecha de referencia.
- **Componente espacial:** existe de forma implícita. Las celdas pertenecen a cuatro áreas silvestres contiguas del norte de Colorado, pero el conjunto no incluye coordenadas geográficas. La única referencia espacial disponible es el área silvestre, que se analiza en la sección 2.7.
- **Codificación:** las dos variables categóricas originales (área silvestre, 4 niveles, y tipo de suelo, 40 niveles) vienen codificadas en formato *one-hot*.

## 1.6 Reserva del conjunto de prueba

Antes de cualquier análisis que pueda influir en el modelado, se reserva el 20 % de los datos como conjunto de prueba. La partición es **estratificada por la variable objetivo** para conservar las proporciones de las siete clases, algo imprescindible con clases que representan menos del 1 % de los datos, y usa la semilla `42`. El conjunto de prueba solo se utiliza en la evaluación final del capítulo 3.

```{code-cell} ipython3
:tags: [hide-input]

particion = pd.DataFrame(
    {
        "Entrenamiento (n)": y_train.value_counts().sort_index(),
        "Prueba (n)": y_test.value_counts().sort_index(),
        "Entrenamiento (%)": (100 * y_train.value_counts(normalize=True).sort_index()).round(2),
        "Prueba (%)": (100 * y_test.value_counts(normalize=True).sort_index()).round(2),
    }
)
particion.index = [f"{k} {CLASS_NAMES[k]}" for k in particion.index]
particion.loc["Total"] = [len(y_train), len(y_test), 100.0, 100.0]
particion = particion.astype({"Entrenamiento (n)": int, "Prueba (n)": int})
particion
```

Las proporciones por clase coinciden en ambos conjuntos hasta la segunda cifra decimal, lo que confirma que la estratificación funcionó. Los diagnósticos que implican decisiones de modelado, como los umbrales de valores atípicos, se calculan desde aquí solo con el conjunto de entrenamiento. Los chequeos estructurales que no conducen a ninguna decisión (dimensiones, tipos, faltantes y rangos válidos) se reportan sobre el conjunto completo.

## 1.7 Tamaño de la muestra

```{code-cell} ipython3
:tags: [hide-input]

n, p = X.shape
conteos = y.value_counts().sort_index()
resumen = pd.Series(
    {
        "Observaciones (n)": f"{n:,}".replace(",", "."),
        "Predictoras (p)": p,
        "Relación n/p": f"{n / p:,.0f}".replace(",", "."),
        "Relación n/p (solo entrenamiento)": f"{len(X_train) / p:,.0f}".replace(",", "."),
        "Clase mayoritaria": f"{CLASS_NAMES[conteos.idxmax()]} ({conteos.max():,})".replace(",", "."),
        "Clase minoritaria": f"{CLASS_NAMES[conteos.idxmin()]} ({conteos.min():,})".replace(",", "."),
        "Razón de desbalance (mayoritaria/minoritaria)": f"{conteos.max() / conteos.min():.1f}",
        "Áreas silvestres (entidades de nivel superior)": 4,
    },
    name="Valor",
)
resumen.to_frame()
```

La relación entre observaciones y predictoras es alta tanto en el conjunto completo (más
de 10.000) como en el de entrenamiento, que es el que de verdad importa para ajustar el
modelo (más de 8.000), de modo que el riesgo de sobreajuste por dimensionalidad es bajo
para un modelo lineal. Aun así, el **tamaño efectivo de la muestra es menor que el número
de filas** por dos motivos:

- **Autocorrelación espacial.** Las celdas vecinas comparten elevación, suelo y cobertura. Aunque el conjunto no trae coordenadas, es razonable suponer que muchas celdas del muestreo son adyacentes y, por tanto, no son observaciones independientes.
- **Pocas entidades de nivel superior.** Solo hay cuatro áreas silvestres, y algunas clases dependen casi por completo de una de ellas (sección 2.1). Para aprender la clase 4 (álamo/sauce) hay 2.747 celdas, todas del área Cache la Poudre, lo que equivale en la práctica a un número pequeño de rodales.

## 1.8 Calidad de los datos

### Valores faltantes

```{code-cell} ipython3
:tags: [hide-input]

faltantes = df.isna().sum()
print(f"Celdas faltantes en todo el conjunto: {int(faltantes.sum())}")
print(f"Columnas con al menos un faltante: {int((faltantes > 0).sum())}")
```

El conjunto **no tiene valores faltantes** en ninguna de sus 55 columnas, lo que coincide con la documentación del repositorio. Por lo tanto, no es necesario analizar el mecanismo de ausencia (MCAR, MAR o MNAR) ni aplicar pruebas como la de Little {cite}`little1988`, y el *pipeline* no incluye imputación. Sí se revisa que no existan valores centinela que oculten faltantes (por ejemplo, ceros o códigos como −999 en lugar de nulos); ese análisis se presenta en la subsección de valores imposibles.

### Duplicados exactos y casi-duplicados

```{code-cell} ipython3
:tags: [hide-input]

dup_full = df.duplicated().sum()
dup_X = X.duplicated().sum()

# Casi-duplicados: celdas idénticas en área, suelo, orientación y pendiente, con diferencias
# menores a la resolución de los datos en las variables continuas.
redondeo = X_train.copy()
redondeo["Elevation"] = (redondeo["Elevation"] / 10).round()
for col in ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]:
    redondeo[col] = (redondeo[col] / 30).round()
redondeo["Vertical_Distance_To_Hydrology"] = (redondeo["Vertical_Distance_To_Hydrology"] / 5).round()
for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
    redondeo[col] = (redondeo[col] / 5).round()

grupos = redondeo.assign(y=y_train).groupby(FEATURES, sort=False)["y"]
tam = grupos.transform("size")
n_clases = grupos.transform("nunique")
casi = tam > 1

pd.Series(
    {
        "Duplicados exactos (filas completas)": dup_full,
        "Duplicados exactos en predictoras": dup_X,
        "Casi-duplicados en entrenamiento (filas)": int(casi.sum()),
        "Casi-duplicados con etiquetas distintas (filas)": int((casi & (n_clases > 1)).sum()),
    },
    name="Filas",
).to_frame()
```

No hay duplicados exactos, ni en las filas completas ni en las predictoras. Para detectar casi-duplicados se redondearon las variables continuas a una resolución cercana a la del dato original (10 m de elevación, 30 m en distancias horizontales, que es el tamaño de la celda, 5 m en distancia vertical y 5 unidades de sombreado) y se buscaron coincidencias en las 54 predictoras. Solo 392 filas de entrenamiento (0,08 %) tienen al menos un casi-duplicado, una cantidad despreciable frente a las 464.809 filas. De ellas, 16 pertenecen a grupos con etiquetas distintas, lo cual es una señal, aunque pequeña, de **ruido irreducible**: sitios cartográficamente casi idénticos con cobertura distinta. Esto fija un techo para cualquier modelo que use solo estas variables. Los casi-duplicados se conservan, porque corresponden a celdas reales y no a registros repetidos.

### Valores imposibles o inconsistentes

```{code-cell} ipython3
:tags: [hide-input]

chequeos = pd.DataFrame(
    [
        ("Aspect fuera de [0, 360]", int(((X["Aspect"] < 0) | (X["Aspect"] > 360)).sum())),
        ("Slope fuera de [0, 90]", int(((X["Slope"] < 0) | (X["Slope"] > 90)).sum())),
        ("Hillshade fuera de [0, 255]", int(((X[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]] < 0) | (X[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]] > 255)).any(axis=1).sum())),
        ("Distancias horizontales negativas", int((X[["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]] < 0).any(axis=1).sum())),
        ("Filas sin exactamente un área silvestre", int((X[WILD_COLS].sum(axis=1) != 1).sum())),
        ("Filas sin exactamente un tipo de suelo", int((X[SOIL_COLS].sum(axis=1) != 1).sum())),
        ("Etiquetas fuera de 1–7", int((~y.isin(range(1, 8))).sum())),
        ("Códigos centinela (−999, −9999)", int(X.isin([-999, -9999]).any(axis=1).sum())),
        ("Distancia vertical negativa (válida)", int((X["Vertical_Distance_To_Hydrology"] < 0).sum())),
        ("Hillshade_3pm igual a 0", int((X["Hillshade_3pm"] == 0).sum())),
        ("Aspect igual a 360 (equivale a 0)", int((X["Aspect"] == 360).sum())),
    ],
    columns=["Chequeo", "Filas"],
).set_index("Chequeo")
chequeos
```

Todas las variables respetan sus rangos físicos, las codificaciones *one-hot* son consistentes (cada celda pertenece a exactamente un área y un tipo de suelo) y no hay códigos centinela. Tres casos merecen comentario:

- **Distancia vertical negativa.** No es un error: indica que la celda está por debajo del nivel de la fuente de agua más cercana.
- **Orientación igual a 360°.** Es la misma dirección que 0° (norte). La discontinuidad artificial entre 359° y 0° se resuelve en el preprocesamiento con la transformación seno-coseno (sección 2.9).
- **Sombreado de las 3 p. m. igual a 0.** Podría parecer un faltante codificado como cero, pero no lo es:

```{code-cell} ipython3
:tags: [hide-input]

ceros = X_train["Hillshade_3pm"] == 0
comparacion = pd.DataFrame(
    {
        "Hillshade_3pm = 0": X_train.loc[ceros, ["Aspect", "Slope", "Hillshade_9am"]].median(),
        "Resto de celdas": X_train.loc[~ceros, ["Aspect", "Slope", "Hillshade_9am"]].median(),
    }
).round(1)
comparacion.index = ["Orientación mediana (°)", "Pendiente mediana (°)", "Sombreado 9 a. m. mediano"]
comparacion
```

Las celdas con sombreado nulo a las 3 p. m. son laderas **empinadas (pendiente mediana cercana a 36°) orientadas al este (orientación mediana cercana a 86°)** y muy iluminadas en la mañana. Es exactamente la geometría que queda en sombra total cuando el sol está al oeste por la tarde, de modo que los ceros son valores físicamente válidos y se conservan.

### Valores atípicos

```{code-cell} ipython3
:tags: [hide-input]

q1 = X_train[NUM_COLS].quantile(0.25)
q3 = X_train[NUM_COLS].quantile(0.75)
iqr = q3 - q1
fuera = (X_train[NUM_COLS] < q1 - 1.5 * iqr) | (X_train[NUM_COLS] > q3 + 1.5 * iqr)
atipicos = pd.DataFrame(
    {
        "Límite inferior": (q1 - 1.5 * iqr).round(1),
        "Límite superior": (q3 + 1.5 * iqr).round(1),
        "Atípicos (n)": fuera.sum(),
        "Atípicos (%)": (100 * fuera.mean()).round(2),
    }
)
atipicos.index = [DISPLAY_NAMES[c] for c in atipicos.index]
atipicos
```

Con la regla de 1,5 veces el rango intercuartílico (calculada solo en entrenamiento), cada variable tiene entre 0 % y 5,4 % de valores atípicos. Estos valores no son errores de medición: corresponden a zonas reales del bosque, como las cotas más bajas del valle del río Cache la Poudre, pendientes muy pronunciadas o celdas alejadas de cursos de agua. Además, varias de estas zonas concentran clases minoritarias (por ejemplo, el álamo/sauce en elevaciones bajas), por lo que eliminarlas sesgaría el aprendizaje justamente de las clases más difíciles. **Decisión:** se conservan todos los valores atípicos; el escalado estándar y la transformación logarítmica de las distancias (sección 2.9) reducen su influencia en la regresión logística. El análisis gráfico y multivariado de atípicos se presenta en las secciones 2.2 y 2.4.

### Sesgos de muestreo y representatividad

- **Cobertura geográfica limitada.** Los datos provienen de cuatro áreas silvestres del Roosevelt National Forest, entre unos 1.860 y 3.860 m de altitud. Un modelo entrenado aquí es representativo de bosques montanos y subalpinos de las Montañas Rocosas del norte de Colorado, no de bosques en general.
- **Áreas con mínima intervención humana.** Las áreas silvestres tienen poca perturbación antrópica, por lo que la cobertura refleja sobre todo procesos ecológicos naturales. Esto favorece la relación entre cartografía y cobertura, pero limita la aplicación del modelo a zonas intervenidas.
- **Tamaños desiguales de las áreas.** Rawah y Comanche Peak aportan cerca del 88 % de las celdas; Neota y Cache la Poudre, el resto. Los patrones de las áreas grandes dominan el ajuste.
- **Dependencia de clases respecto al área.** El álamo/sauce aparece solo en Cache la Poudre, y el pino ponderosa y el abeto de Douglas se concentran allí. El modelo puede aprender el área como atajo para esas clases, lo que se analiza en la sección 2.7.
- **Etiquetas derivadas de un sistema de información.** La cobertura proviene del sistema de información de recursos del USFS, no de un inventario de campo celda por celda, por lo que las etiquetas pueden contener errores de clasificación.

## 1.9 Consideraciones éticas

El conjunto no contiene datos personales ni información sobre individuos, de modo que no existen riesgos de reidentificación ni se requiere anonimización. Las celdas no tienen coordenadas geográficas precisas, lo que además impide ubicar con exactitud recursos sensibles (por ejemplo, hábitats de especies protegidas). Los datos son públicos y se usan bajo licencia CC BY 4.0, con la debida atribución.

El principal riesgo ético es de **uso indebido de las predicciones**: un modelo con errores sistemáticos en las clases minoritarias podría llevar a subestimar ecosistemas raros, como los bosques ribereños de álamo y sauce, en decisiones de manejo o conservación. Por eso el documento reporta métricas por clase, y no solo exactitud global.
