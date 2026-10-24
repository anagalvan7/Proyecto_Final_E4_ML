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

# 3. Modelo base

Este capítulo construye la referencia de desempeño del estudio. Se compara una **línea base trivial** (`DummyClassifier`) con un **modelo base** de regresión logística multinomial entrenado {cite}`hosmer2013` dentro de un `Pipeline`, se eligen sus hiperparámetros con validación cruzada estratificada y se evalúa una única vez sobre el conjunto de prueba reservado en la sección 1.6.

```{code-cell} ipython3
:tags: [hide-input]

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, StratifiedKFold, cross_validate, learning_curve,
)
from sklearn.preprocessing import label_binarize

from _utils import (
    CLASS_NAMES, CLASS_PALETTE, CLASS_SHORT, DISPLAY_NAMES, ENG_NUM_COLS, RANDOM_STATE,
    SOIL_COLS, WILD_COLS, WILD_NAMES, build_logreg, build_preprocessor, load_covertype,
    split_covertype, stratified_sample, wilderness_label,
)

sns.set_theme(style="whitegrid", context="notebook")
CLASES = list(range(1, 8))

df, X, y = load_covertype()
X_train, X_test, y_train, y_test = split_covertype(X, y)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

## 3.1 Variable objetivo y esquema de evaluación

- **Variable objetivo:** `Cover_Type`, tipo de cobertura forestal dominante, con siete clases nominales (sección 1.4).
- **Partición:** 80 % entrenamiento y 20 % prueba, estratificada por clase y con semilla 42, definida antes de cualquier preprocesamiento (sección 1.6).
- **Validación cruzada:** `StratifiedKFold` con 5 pliegues, del mismo tipo que la partición. Como prueba de sensibilidad espacial se usa además `GroupKFold` por área silvestre (sección 3.8).
- **Métrica de selección:** F1 macro, porque da el mismo peso a las siete clases y no premia al modelo que ignora las minoritarias (sección 2.1). Se reportan también *accuracy*, exactitud balanceada, precisión y *recall* macro, F1 ponderado, AUC uno contra el resto y log loss.

## 3.2 Línea base trivial

Se evalúan tres estrategias de `DummyClassifier`: predecir siempre la clase más frecuente, predecir al azar respetando las proporciones de clase (estratificada) y predecir al azar con probabilidad uniforme.

```{code-cell} ipython3
:tags: [hide-input]

def metricas(y_true, pred, proba):
    """Métricas de clasificación multiclase usadas en todo el capítulo."""
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Exactitud balanceada": balanced_accuracy_score(y_true, pred),
        "Precisión macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1 ponderado": f1_score(y_true, pred, average="weighted", zero_division=0),
        "AUC OvR macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "Log loss": log_loss(y_true, proba, labels=CLASES),
    }

dummies = {
    "Dummy (más frecuente)": DummyClassifier(strategy="most_frequent"),
    "Dummy (estratificado)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    "Dummy (uniforme)": DummyClassifier(strategy="uniform", random_state=RANDOM_STATE),
}
res_test = {}
for nombre, modelo in dummies.items():
    modelo.fit(X_train, y_train)
    res_test[nombre] = metricas(y_test, modelo.predict(X_test), modelo.predict_proba(X_test))
pd.DataFrame(res_test).T.round(3)
```

El clasificador que siempre predice pino lodgepole alcanza cerca del 49 % de *accuracy*, pero su F1 macro es de apenas 0,09 y su exactitud balanceada es exactamente 1/7 (0,143), porque acierta en una clase y falla en las otras seis. Esta es la referencia mínima que cualquier modelo útil debe superar con claridad, sobre todo en las métricas que ponderan todas las clases por igual.

## 3.3 Selección de hiperparámetros

Se busca en una rejilla la mejor combinación de:

- **Preprocesamiento:** básico (solo escalado) o con ingeniería de variables (sección 2.9).
- **Regularización:** inverso de la fuerza de regularización L2, $C \in \{0{,}1;\ 1;\ 10\}$.
- **Ponderación de clases:** sin ponderación o `class_weight="balanced"`.

Cada una de las 12 combinaciones se evalúa con validación cruzada estratificada de 5 pliegues. Ajustar una regresión logística sobre las 464.809 celdas de entrenamiento toma varios minutos, así que la búsqueda se hace sobre una **submuestra estratificada de 30.000 celdas** del entrenamiento. La curva de aprendizaje de la sección 3.9 muestra que el desempeño de la regresión logística ya está prácticamente estabilizado con ese tamaño de muestra, de modo que la submuestra no cambia la elección de hiperparámetros. El modelo final sí se entrena con todo el conjunto de entrenamiento.

```{code-cell} ipython3
:tags: [hide-input]

X_sub, y_sub = stratified_sample(X_train, y_train, 30000)
rejilla = {
    "prep": [build_preprocessor(engineered=False), build_preprocessor(engineered=True)],
    "clf__C": [0.1, 1.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}
busqueda = GridSearchCV(
    build_logreg(),
    rejilla,
    scoring={"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
    refit=False,
    cv=cv5,
    n_jobs=-1,
)
inicio = time.time()
busqueda.fit(X_sub, y_sub)
print(f"Tiempo de la búsqueda: {time.time() - inicio:.0f} s")

cvr = pd.DataFrame(busqueda.cv_results_)
cvr["Preprocesamiento"] = np.where(cvr["param_prep"].astype(str).str.startswith("Pipeline"), "Con ingeniería", "Básico")
cvr["Ponderación"] = cvr["param_clf__class_weight"].fillna("ninguna").astype(str)
tabla_cv = cvr[["Preprocesamiento", "param_clf__C", "Ponderación", "mean_test_f1_macro", "std_test_f1_macro",
                "mean_test_accuracy", "mean_test_balanced_accuracy"]].rename(columns={
    "param_clf__C": "C", "mean_test_f1_macro": "F1 macro (media)", "std_test_f1_macro": "F1 macro (desv.)",
    "mean_test_accuracy": "Accuracy (media)", "mean_test_balanced_accuracy": "Exactitud balanceada (media)",
}).sort_values("F1 macro (media)", ascending=False).reset_index(drop=True)
tabla_cv.round(4)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pond in zip(axes, ["ninguna", "balanced"]):
    piv = tabla_cv[tabla_cv["Ponderación"] == pond].pivot(index="Preprocesamiento", columns="C", values="F1 macro (media)")
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Greens", ax=ax, vmin=tabla_cv["F1 macro (media)"].min(),
                vmax=tabla_cv["F1 macro (media)"].max())
    ax.set_title(f"F1 macro en validación cruzada (ponderación: {pond})")
plt.tight_layout()
plt.show()

mejor = tabla_cv.iloc[0]
MEJOR_ENG = mejor["Preprocesamiento"] == "Con ingeniería"
MEJOR_C = float(mejor["C"])
MEJOR_CW = None if mejor["Ponderación"] == "ninguna" else "balanced"
print(f"Mejor configuración: preprocesamiento {mejor['Preprocesamiento'].lower()}, C = {MEJOR_C}, "
      f"ponderación = {mejor['Ponderación']} (F1 macro = {mejor['F1 macro (media)']:.3f} ± {mejor['F1 macro (desv.)']:.3f})")
```

Tres conclusiones se desprenden de la búsqueda:

1. **La ingeniería de variables mejora el modelo** en todas las combinaciones de $C$ y ponderación. La representación circular de la orientación, el logaritmo de las distancias y las dos variables nuevas de posición respecto al agua aportan información que la regresión logística no puede construir por sí sola, porque es lineal en sus entradas.
2. **La regularización tiene poco efecto** para $C \geq 1$: con más de 8.000 observaciones de entrenamiento por predictora (sección 1.7), la varianza del estimador es baja y la penalización deja de ser determinante. Solo la regularización más fuerte ($C = 0{,}1$) reduce un poco el desempeño.
3. **La ponderación de clases implica un compromiso claro.** Con `class_weight="balanced"` la exactitud balanceada sube de forma notable, porque el modelo detecta mucho mejor las clases minoritarias, pero el *accuracy* y el F1 macro bajan, porque el modelo empieza a asignar muchas celdas de las clases mayoritarias a las minoritarias y su precisión en estas cae. Según el criterio de selección definido (F1 macro), el modelo base se entrena sin ponderación; la versión ponderada se evalúa también en prueba (sección 3.6) para documentar el compromiso.

## 3.4 Entrenamiento del modelo final

```{code-cell} ipython3
:tags: [hide-input]

inicio = time.time()
modelo = build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, class_weight=MEJOR_CW, max_iter=3000)
modelo.fit(X_train, y_train)
print(f"Tiempo de entrenamiento con {len(X_train):,} celdas: {time.time() - inicio:.0f} s".replace(",", "."))
print(f"Iteraciones del optimizador: {modelo.named_steps['clf'].n_iter_[0]} (máximo 3000)")

X_cv, y_cv = stratified_sample(X_train, y_train, 100000)
cv_final = cross_validate(
    build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, class_weight=MEJOR_CW, max_iter=3000),
    X_cv, y_cv, cv=cv5, n_jobs=-1, return_train_score=True,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
)
resumen_cv = pd.DataFrame(
    {
        "Entrenamiento (media)": [cv_final[f"train_{m}"].mean() for m in ["accuracy", "f1_macro", "balanced_accuracy"]],
        "Validación (media)": [cv_final[f"test_{m}"].mean() for m in ["accuracy", "f1_macro", "balanced_accuracy"]],
        "Validación (desv. est.)": [cv_final[f"test_{m}"].std() for m in ["accuracy", "f1_macro", "balanced_accuracy"]],
    },
    index=["Accuracy", "F1 macro", "Exactitud balanceada"],
).round(4)
resumen_cv
```

La tabla muestra la validación cruzada estratificada de 5 pliegues del modelo elegido sobre una submuestra de 100.000 celdas. Las métricas de entrenamiento y validación son prácticamente iguales y la desviación estándar entre pliegues es muy pequeña, lo que indica un modelo **estable y sin sobreajuste**. Si el modelo tiene alguna limitación, es de sesgo (su forma lineal es demasiado simple), no de varianza.

## 3.5 Evaluación en el conjunto de prueba

### Métricas globales con intervalos de confianza

Los intervalos de confianza del 95 % se obtienen con *bootstrap* percentil: se remuestrean con reemplazo las 116.203 celdas de prueba 200 veces y se recalcula cada métrica.

```{code-cell} ipython3
:tags: [hide-input]

pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)
res_test["Regresión logística"] = metricas(y_test, pred, proba)

rng = np.random.default_rng(RANDOM_STATE)
yt = y_test.to_numpy()
boot = {m: [] for m in ["Accuracy", "Exactitud balanceada", "F1 macro", "AUC OvR macro"]}
for _ in range(200):
    i = rng.integers(0, len(yt), len(yt))
    boot["Accuracy"].append(accuracy_score(yt[i], pred[i]))
    boot["Exactitud balanceada"].append(balanced_accuracy_score(yt[i], pred[i]))
    boot["F1 macro"].append(f1_score(yt[i], pred[i], average="macro"))
    boot["AUC OvR macro"].append(roc_auc_score(yt[i], proba[i], multi_class="ovr", average="macro"))
ic = pd.DataFrame(
    {
        "Estimación": [res_test["Regresión logística"][m] for m in boot],
        "IC 95 % inferior": [np.percentile(v, 2.5) for v in boot.values()],
        "IC 95 % superior": [np.percentile(v, 97.5) for v in boot.values()],
    },
    index=list(boot),
).round(4)
ic
```

```{code-cell} ipython3
:tags: [hide-input]

comparacion = pd.DataFrame(res_test).T[
    ["Accuracy", "Exactitud balanceada", "Precisión macro", "Recall macro", "F1 macro", "F1 ponderado", "AUC OvR macro", "Log loss"]
]
comparacion.round(3)
```

```{code-cell} ipython3
:tags: [hide-input]

base = res_test["Dummy (más frecuente)"]
lr = res_test["Regresión logística"]
mejoras = pd.DataFrame(
    {
        "Dummy (más frecuente)": [base[m] for m in ["Accuracy", "Exactitud balanceada", "F1 macro", "AUC OvR macro"]],
        "Regresión logística": [lr[m] for m in ["Accuracy", "Exactitud balanceada", "F1 macro", "AUC OvR macro"]],
    },
    index=["Accuracy", "Exactitud balanceada", "F1 macro", "AUC OvR macro"],
)
mejoras["Mejora absoluta"] = mejoras["Regresión logística"] - mejoras["Dummy (más frecuente)"]
mejoras.round(3)
```

La regresión logística **supera ampliamente a todas las líneas base triviales** en todas las métricas: gana 23,7 puntos de *accuracy* (0,724 frente a 0,488), 38,9 puntos de exactitud balanceada y 46,4 puntos de F1 macro sobre el clasificador de la clase más frecuente. Los intervalos de confianza son estrechos (alrededor de medio punto porcentual de amplitud para el *accuracy*) gracias al tamaño del conjunto de prueba. Las métricas de prueba coinciden con las de validación cruzada (0,727 de *accuracy* y 0,565 de F1 macro), lo que confirma que la estimación no está sesgada por la selección de hiperparámetros.

Sin embargo, la brecha entre el *accuracy* (0,72) y la exactitud balanceada o el F1 macro (0,53 y 0,56) revela que **el buen desempeño global descansa sobre las clases mayoritarias**. El AUC uno contra el resto es alto (0,94), lo que indica que el modelo ordena bien las celdas según su probabilidad de pertenecer a cada clase, aunque la regla de asignar la clase con mayor probabilidad favorece a las clases frecuentes.

### Desempeño por clase

Los intervalos de confianza del 95 % para las métricas por clase se obtienen con el
mismo procedimiento de *bootstrap* percentil (200 remuestreos) usado en las métricas
globales.

```{code-cell} ipython3
:tags: [hide-input]

reporte = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T
reporte = reporte.loc[[str(k) for k in CLASES]]
reporte.index = [CLASS_SHORT[k] for k in CLASES]
reporte.columns = ["Precisión", "Recall", "F1", "Soporte"]
reporte["Soporte"] = reporte["Soporte"].astype(int)

rng_clase = np.random.default_rng(RANDOM_STATE)
n_test = len(yt)
boot_clase = {k: {"Precisión": [], "Recall": [], "F1": []} for k in CLASES}
for _ in range(200):
    i = rng_clase.integers(0, n_test, n_test)
    p_i = precision_score(yt[i], pred[i], average=None, labels=CLASES, zero_division=0)
    r_i = recall_score(yt[i], pred[i], average=None, labels=CLASES, zero_division=0)
    f_i = f1_score(yt[i], pred[i], average=None, labels=CLASES, zero_division=0)
    for idx, k in enumerate(CLASES):
        boot_clase[k]["Precisión"].append(p_i[idx])
        boot_clase[k]["Recall"].append(r_i[idx])
        boot_clase[k]["F1"].append(f_i[idx])

def _fmt_ic(punto, muestras):
    lo, hi = np.percentile(muestras, [2.5, 97.5])
    return f"{punto:.3f} [{lo:.3f}, {hi:.3f}]"

filas_ic = []
for k in CLASES:
    nombre = CLASS_SHORT[k]
    filas_ic.append({
        "Precisión": _fmt_ic(reporte.loc[nombre, "Precisión"], boot_clase[k]["Precisión"]),
        "Recall": _fmt_ic(reporte.loc[nombre, "Recall"], boot_clase[k]["Recall"]),
        "F1": _fmt_ic(reporte.loc[nombre, "F1"], boot_clase[k]["F1"]),
        "Soporte": reporte.loc[nombre, "Soporte"],
    })
reporte_ic = pd.DataFrame(filas_ic, index=[CLASS_SHORT[k] for k in CLASES])
reporte_ic
```

Las clases mayoritarias (pino lodgepole, abeto/pícea) tienen intervalos muy estrechos
gracias a su gran soporte; las minoritarias (álamo temblón, álamo/sauce) tienen
intervalos bastante más amplios, lo que es consistente con un tamaño de muestra efectivo
mucho menor para esas clases.

### Matriz de confusión

```{code-cell} ipython3
:tags: [hide-input]

cm = confusion_matrix(y_test, pred, labels=CLASES)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
etiquetas = [CLASS_SHORT[k] for k in CLASES]
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0], xticklabels=etiquetas, yticklabels=etiquetas,
            cbar=False, annot_kws={"size": 8})
axes[0].set_title("Matriz de confusión (conteos)")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], xticklabels=etiquetas, yticklabels=etiquetas,
            vmin=0, vmax=1, annot_kws={"size": 8})
axes[1].set_title("Matriz de confusión normalizada por fila (recall)")
for ax in axes:
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
```

La matriz de confusión identifica cuatro patrones de error:

- **Confusión entre abeto/pícea y pino lodgepole.** Es el error más frecuente en términos absolutos: el 29 % del abeto/pícea se clasifica como pino lodgepole y el 18 % del pino lodgepole como abeto/pícea (más de 22.000 celdas entre ambos sentidos). Las dos clases comparten elevaciones, suelos y áreas (sección 2.3), y el modelo lineal no encuentra una frontera que las separe bien.
- **Confusión en el límite superior del bosque.** El 43 % del *krummholz* se clasifica como abeto/pícea, la especie con la que comparte la franja más alta.
- **Confusión dentro del grupo de zonas bajas.** El 46 % del abeto de Douglas se asigna al pino ponderosa, y el 36 % del álamo/sauce también. Las tres clases comparten el área de Cache la Poudre, las elevaciones bajas y suelos similares.
- **El álamo temblón (clase 5) prácticamente no se reconoce.** Solo el 4 % de sus celdas se clasifica bien; el 91 % se asigna al pino lodgepole. Es una clase minoritaria cuyo rango de variables se superpone con el de una clase mayoritaria, el peor escenario para un modelo lineal sin ponderación.

### Curvas ROC y precisión-*recall*

```{code-cell} ipython3
:tags: [hide-input]

Y_bin = label_binarize(y_test, classes=CLASES)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for j, k in enumerate(CLASES):
    fpr, tpr, _ = roc_curve(Y_bin[:, j], proba[:, j])
    auc_k = roc_auc_score(Y_bin[:, j], proba[:, j])
    axes[0].plot(fpr, tpr, color=CLASS_PALETTE[k], lw=1.6, label=f"{CLASS_SHORT[k]} (AUC = {auc_k:.3f})")
    prec, rec, _ = precision_recall_curve(Y_bin[:, j], proba[:, j])
    ap = average_precision_score(Y_bin[:, j], proba[:, j])
    axes[1].plot(rec, prec, color=CLASS_PALETTE[k], lw=1.6, label=f"{CLASS_SHORT[k]} (AP = {ap:.3f}; base = {Y_bin[:, j].mean():.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("Tasa de falsos positivos")
axes[0].set_ylabel("Tasa de verdaderos positivos")
axes[0].set_title("Curvas ROC uno contra el resto")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precisión")
axes[1].set_title("Curvas precisión-recall uno contra el resto")
axes[1].legend(fontsize=7.5, loc="upper right")
plt.tight_layout()
plt.show()
```

Las curvas ROC son altas para todas las clases: el AUC va de 0,85 (pino lodgepole) a 0,997 (álamo/sauce). Paradójicamente, las clases mayoritarias tienen el AUC más bajo, porque son las que más se confunden entre sí. Pero con clases muy desbalanceadas la curva ROC es optimista, ya que la tasa de falsos positivos se calcula sobre una clase negativa enorme. Las **curvas precisión-*recall*** dan una imagen más realista:

- La precisión promedio (AP) de todas las clases supera con amplitud su prevalencia, que es la AP de un clasificador aleatorio. Por ejemplo, el álamo/sauce tiene una AP de 0,66 con una prevalencia de 0,005, lo que confirma que el modelo sí contiene señal para las clases minoritarias.
- El álamo temblón (AP = 0,22) y el abeto de Douglas (AP = 0,45) son los casos más difíciles: su precisión cae rápidamente al aumentar el *recall*, de modo que detectarlos exige aceptar muchos falsos positivos.

### Calibración

Un modelo está bien calibrado si, entre las celdas a las que asigna una probabilidad de 0,7 para una clase, cerca del 70 % pertenecen realmente a ella. Se evalúa con curvas de confiabilidad uno contra el resto (10 intervalos por cuantiles de la probabilidad predicha) y con la puntuación de Brier multiclase {cite}`brier1950`, que es el error cuadrático medio entre el vector de probabilidades y el vector *one-hot* de la clase real.

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
brier = {}
for j, k in enumerate(CLASES):
    frac, media = calibration_curve(Y_bin[:, j], proba[:, j], n_bins=10, strategy="quantile")
    axes[0].plot(media, frac, "o-", color=CLASS_PALETTE[k], ms=3, lw=1.3, label=CLASS_SHORT[k])
    brier[CLASS_SHORT[k]] = np.mean((proba[:, j] - Y_bin[:, j]) ** 2)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("Probabilidad predicha (media del intervalo)")
axes[0].set_ylabel("Frecuencia observada")
axes[0].set_title("Curvas de confiabilidad uno contra el resto")
axes[0].legend(fontsize=8)

confianza = proba.max(axis=1)
acierto = (pred == yt).astype(int)
frac_top, media_top = calibration_curve(acierto, confianza, n_bins=10, strategy="quantile")
axes[1].plot(media_top, frac_top, "o-", color="#1b6f5c", label="Clase predicha")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("Probabilidad de la clase predicha")
axes[1].set_ylabel("Proporción de aciertos")
axes[1].set_title("Calibración de la clase predicha (top-label)")
axes[1].legend()
plt.tight_layout()
plt.show()

brier_multi = np.mean(np.sum((proba - Y_bin) ** 2, axis=1))
brier_dummy = np.mean(np.sum((dummies["Dummy (estratificado)"].predict_proba(X_test) - Y_bin) ** 2, axis=1))
prior = y_train.value_counts(normalize=True).sort_index().to_numpy()
brier_prior = np.mean(np.sum((prior - Y_bin) ** 2, axis=1))
print(f"Brier multiclase de la regresión logística: {brier_multi:.4f}")
print(f"Brier multiclase de predecir siempre las proporciones de clase: {brier_prior:.4f}")
pd.Series(brier, name="Brier uno contra el resto").round(4).to_frame().T
```

Las curvas de confiabilidad están prácticamente sobre la diagonal, tanto por clase como para la clase predicha, lo que indica que **el modelo está muy bien calibrado**: cuando asigna una probabilidad de 0,8 a su predicción, acierta en cerca del 80 % de los casos. Para las clases minoritarias las probabilidades predichas rara vez superan 0,3, por eso sus curvas cubren un rango más corto, pero dentro de ese rango también siguen la diagonal. El Brier multiclase (0,381) mejora de forma sustancial respecto a la referencia que asigna a cada celda las proporciones de clase (0,623). La buena calibración es una propiedad conocida de la regresión logística, que optimiza directamente la verosimilitud.

## 3.6 Efecto de la ponderación de clases

Para documentar el compromiso observado en la búsqueda de hiperparámetros, se entrena la misma configuración con `class_weight="balanced"` sobre todo el entrenamiento y se evalúa en prueba.

```{code-cell} ipython3
:tags: [hide-input]

modelo_bal = build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, class_weight="balanced", max_iter=3000)
modelo_bal.fit(X_train, y_train)
pred_bal = modelo_bal.predict(X_test)
res_bal = metricas(y_test, pred_bal, modelo_bal.predict_proba(X_test))

comp_pond = pd.DataFrame({"Sin ponderación": res_test["Regresión logística"], "Ponderación balanceada": res_bal}).T
display(comp_pond[["Accuracy", "Exactitud balanceada", "F1 macro", "AUC OvR macro", "Log loss"]].round(3))

recall_clases = pd.DataFrame(
    {
        "Recall sin ponderación": recall_score(y_test, pred, average=None, labels=CLASES),
        "Recall con ponderación": recall_score(y_test, pred_bal, average=None, labels=CLASES),
        "Precisión sin ponderación": precision_score(y_test, pred, average=None, labels=CLASES, zero_division=0),
        "Precisión con ponderación": precision_score(y_test, pred_bal, average=None, labels=CLASES, zero_division=0),
    },
    index=[CLASS_SHORT[k] for k in CLASES],
).round(3)
recall_clases
```

La ponderación balanceada **eleva el *recall* de las clases minoritarias**: el del álamo temblón pasa de 0,04 a 0,79, el del álamo/sauce de 0,53 a 0,93 y el del *krummholz* de 0,56 a 0,90. El costo es una **caída de su precisión** (la del álamo temblón baja de 0,49 a 0,11) y del *recall* del pino lodgepole (de 0,80 a 0,53). En conjunto, la exactitud balanceada sube de 0,53 a 0,73, pero el *accuracy* baja de 0,72 a 0,60 y el F1 macro, de 0,56 a 0,53. La elección entre ambas versiones depende del uso: si el objetivo es inventariar ecosistemas raros, donde omitir un rodal es más costoso que revisar una falsa alarma, la versión ponderada es preferible; si el objetivo es la exactitud cartográfica global, lo es la versión sin ponderación. La Parte II resuelve esta disyuntiva adoptando el F1 macro como criterio de selección (sección 5.1) y comparando de forma sistemática cuatro estrategias de balanceo en siete algoritmos.

## 3.7 Diagnóstico de errores según la estructura de los datos

Los datos no tienen componente temporal ni coordenadas, así que no aplican el análisis de autocorrelación de residuos (ACF) ni el I de Moran de los residuos. El diagnóstico equivalente disponible es revisar si los errores se concentran en ciertas zonas (área silvestre) o en ciertos rangos del principal gradiente ambiental (elevación). Una concentración sistemática indicaría información estructural que el modelo no está capturando.

```{code-cell} ipython3
:tags: [hide-input]

area_test = wilderness_label(X_test)
errores = pd.DataFrame({"Area": area_test.values, "Elevacion": X_test["Elevation"].values,
                        "Real": yt, "Pred": pred})
errores["Acierto"] = errores["Real"] == errores["Pred"]
por_area = errores.groupby("Area").apply(
    lambda g: pd.Series({"Celdas de prueba": len(g), "Accuracy": g["Acierto"].mean(),
                         "F1 macro (clases presentes)": f1_score(g["Real"], g["Pred"], average="macro",
                                                                labels=sorted(g["Real"].unique()), zero_division=0)})
).loc[list(WILD_NAMES.values())]
por_area["Celdas de prueba"] = por_area["Celdas de prueba"].astype(int)

errores["Franja de elevación"] = pd.cut(errores["Elevacion"], bins=range(1800, 4000, 200))
por_elev = errores.groupby("Franja de elevación").agg(Celdas=("Acierto", "size"), Accuracy=("Acierto", "mean"))
por_elev = por_elev[por_elev["Celdas"] > 50]

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(por_elev.index.astype(str), por_elev["Accuracy"], color="#3b7a57")
ax.axhline(errores["Acierto"].mean(), color="#b0413e", ls="--", lw=1, label="Accuracy global")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Franja de elevación (m)")
ax.set_title("Accuracy del modelo por franja de elevación (conjunto de prueba)")
ax.tick_params(axis="x", rotation=40)
ax.legend()
plt.tight_layout()
plt.show()
por_area.round(3)
```

Los errores **no se distribuyen de forma homogénea**:

- **Por área**, el *accuracy* va de 0,76 en Rawah a 0,63 en Neota y Cache la Poudre. Las áreas pequeñas, con menos peso en el ajuste, tienen peor desempeño.
- **Por elevación**, el *accuracy* es más alto entre 2.600 y 3.000 m (0,75–0,78), donde domina el pino lodgepole, y más bajo en los extremos del gradiente: 0,62–0,65 por debajo de 2.400 m, donde se mezclan el pino ponderosa, el abeto de Douglas y el álamo/sauce, y 0,64 por encima de 3.600 m, donde el *krummholz* se confunde con el abeto/pícea.

Este patrón es el equivalente, en estos datos, a residuos con dependencia estructural: el modelo comete errores sistemáticos en ciertas zonas y franjas de altitud, lo que indica que el efecto de las variables cambia según la ubicación y requeriría interacciones o un modelo no lineal.

## 3.8 Sensibilidad a la partición: generalización a áreas no vistas

Como prueba de sensibilidad espacial (sección 2.7), se evalúa el modelo con `GroupKFold` dejando fuera un área silvestre completa en cada pliegue. Por costo computacional, se usa una submuestra estratificada de 60.000 celdas del entrenamiento.

```{code-cell} ipython3
:tags: [hide-input]

X_g, y_g = stratified_sample(X_train, y_train, 60000)
grupos = wilderness_label(X_g)
filas = []
modelo_g = build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, class_weight=MEJOR_CW, max_iter=3000)
for tr, te in GroupKFold(n_splits=4).split(X_g, y_g, grupos):
    area_fuera = grupos.iloc[te].iloc[0]
    modelo_g.fit(X_g.iloc[tr], y_g.iloc[tr])
    p = modelo_g.predict(X_g.iloc[te])
    clases_no_vistas = sorted(set(y_g.iloc[te]) - set(y_g.iloc[tr]))
    filas.append({
        "Área evaluada (no vista)": area_fuera,
        "Celdas": len(te),
        "Accuracy": accuracy_score(y_g.iloc[te], p),
        "F1 macro (clases presentes)": f1_score(y_g.iloc[te], p, average="macro",
                                                 labels=sorted(y_g.iloc[te].unique()), zero_division=0),
        "Clases del área nunca vistas en entrenamiento": ", ".join(CLASS_SHORT[k] for k in clases_no_vistas) or "Ninguna",
    })
loao = pd.DataFrame(filas).set_index("Área evaluada (no vista)").round(3)
display(loao)
print(f"Accuracy promedio dejando un área fuera: {loao['Accuracy'].mean():.3f} "
      f"(frente a {res_test['Regresión logística']['Accuracy']:.3f} con partición estratificada aleatoria)")
```

Cuando el modelo debe predecir un área que nunca vio, el *accuracy* promedio cae de 0,72 a 0,56, y el descenso afecta a todas las áreas: el valor va de 0,58 en Rawah a 0,53 en Cache la Poudre, con un F1 macro de las clases presentes entre 0,37 y 0,43. Cache la Poudre es el caso extremo: al dejar fuera Cache la Poudre, el modelo nunca ve el álamo/sauce, que solo existe allí, y tampoco ve la mayor parte de los ejemplos de pino ponderosa y abeto de Douglas. Esto confirma dos cosas:

1. La partición estratificada aleatoria mide bien el desempeño **dentro de las áreas muestreadas**, que es el objetivo del proyecto, pero es optimista para zonas geográficas nuevas.
2. El modelo aprovecha el área silvestre y los suelos específicos de cada área como información geográfica. Para usarlo fuera del Roosevelt National Forest haría falta muestrear otras regiones.

## 3.9 Curva de aprendizaje

```{code-cell} ipython3
:tags: [hide-input]

X_lc, y_lc = stratified_sample(X_train, y_train, 120000)
tamanos, sc_train, sc_val = learning_curve(
    build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, class_weight=MEJOR_CW, max_iter=3000),
    X_lc, y_lc, train_sizes=[1000, 3000, 10000, 30000, 60000, 90000],
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1_macro", n_jobs=-1, shuffle=True, random_state=RANDOM_STATE,
)
fig, ax = plt.subplots(figsize=(9, 4.5))
for sc, nombre, color in [(sc_train, "Entrenamiento", "#1b6f5c"), (sc_val, "Validación", "#b0413e")]:
    ax.plot(tamanos, sc.mean(axis=1), "o-", color=color, label=nombre)
    ax.fill_between(tamanos, sc.mean(axis=1) - sc.std(axis=1), sc.mean(axis=1) + sc.std(axis=1), color=color, alpha=0.15)
ax.set_xscale("log")
ax.set_xlabel("Número de celdas de entrenamiento (escala logarítmica)")
ax.set_ylabel("F1 macro")
ax.set_title("Curva de aprendizaje de la regresión logística")
ax.legend()
plt.tight_layout()
plt.show()
pd.DataFrame({"Tamaño": tamanos, "F1 entrenamiento": sc_train.mean(axis=1),
              "F1 validación": sc_val.mean(axis=1)}).set_index("Tamaño").round(4).T
```

Con 1.000 observaciones el modelo sobreajusta (F1 de 0,71 en entrenamiento frente a 0,51 en validación), pero las dos curvas convergen rápidamente y a partir de unas 30.000 celdas son prácticamente planas y coincidentes (0,570 frente a 0,564). Dos conclusiones:

- **El tamaño de muestra es más que suficiente** para la regresión logística: agregar datos no mejoraría el modelo. Esto justifica también haber hecho la búsqueda de hiperparámetros con una submuestra de 30.000 celdas.
- **No hay sobreajuste**; la limitación del modelo es de **sesgo**. Las dos curvas convergen en un F1 macro cercano a 0,56, muy por debajo de lo que permitiría la señal disponible (sección 3.11). Para mejorar se necesitan modelos más flexibles, no más datos.

## 3.10 Interpretación de los coeficientes

En la regresión logística multinomial cada clase tiene su propio vector de coeficientes. Como las variables numéricas están estandarizadas, cada coeficiente numérico indica cuánto cambia el logaritmo de la razón de probabilidades (*log-odds*) de la clase cuando la variable aumenta una desviación estándar, manteniendo las demás constantes. Las indicadoras de área y suelo se interpretan respecto al promedio de las demás categorías, porque se incluyen todas sin categoría de referencia.

```{code-cell} ipython3
:tags: [hide-input]

clf = modelo.named_steps["clf"]
nombres = modelo.named_steps["prep"][-1].get_feature_names_out() if MEJOR_ENG else modelo.named_steps["prep"].get_feature_names_out()
nombres = [n.split("__", 1)[1] for n in nombres]
coef = pd.DataFrame(clf.coef_.T, index=nombres, columns=[CLASS_SHORT[k] for k in clf.classes_])

numericas = [c for c in coef.index if c not in SOIL_COLS + WILD_COLS]
nombres_legibles = {**DISPLAY_NAMES, "Aspect_sin": "sen(orientación)", "Aspect_cos": "cos(orientación)",
                    "Dist_Hydrology_Euclid": "Dist. euclidiana a agua", "Elevation_Minus_VDH": "Elevación relativa al agua"}
if MEJOR_ENG:
    nombres_legibles.update({c: f"log {DISPLAY_NAMES[c]}" for c in ["Horizontal_Distance_To_Hydrology",
                             "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]})

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
lim = np.abs(coef.loc[numericas]).max().max()
sns.heatmap(coef.loc[numericas].rename(index=nombres_legibles), annot=True, fmt=".1f", cmap="RdBu_r",
            center=0, vmin=-lim, vmax=lim, ax=axes[0], annot_kws={"size": 8})
axes[0].set_title("Coeficientes de las variables numéricas (log-odds por desviación estándar)")
areas_coef = coef.loc[WILD_COLS].rename(index={c: WILD_NAMES[i + 1] for i, c in enumerate(WILD_COLS)})
lim2 = np.abs(areas_coef).max().max()
sns.heatmap(areas_coef, annot=True, fmt=".1f", cmap="RdBu_r", center=0, vmin=-lim2, vmax=lim2, ax=axes[1], annot_kws={"size": 8})
axes[1].set_title("Coeficientes del área silvestre")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

# Un aumento de 100 m en la altitud del terreno, con la distancia vertical al agua fija,
# incrementa en 100 m tanto la elevación como la elevación relativa al agua. Como ambas
# están casi perfectamente correlacionadas, el efecto interpretable es el conjunto.
escalador = modelo.named_steps["prep"][-1].named_transformers_["num"]
sd = dict(zip(escalador.feature_names_in_, escalador.scale_))
b_elev = coef.loc["Elevation"]
b_rel = coef.loc["Elevation_Minus_VDH"] if "Elevation_Minus_VDH" in coef.index else 0 * b_elev
log_odds_100 = b_elev * 100 / sd["Elevation"] + (b_rel * 100 / sd["Elevation_Minus_VDH"] if "Elevation_Minus_VDH" in sd else 0)
pd.DataFrame(
    {
        "Coef. elevación (por 1 DE)": b_elev.round(2),
        "Coef. elevación relativa (por 1 DE)": b_rel.round(2),
        "Razón de odds aislada por +100 m (solo elevación)": np.exp(b_elev * 100 / sd["Elevation"]).round(3),
        "Razón de odds conjunta por +100 m": np.exp(log_odds_100).round(3),
    }
)
```

**Interpretación:**

- **La elevación domina el modelo** y reproduce el gradiente altitudinal, pero su
  coeficiente no puede leerse de forma aislada. La elevación relativa al agua está
  correlacionada casi por completo con ella (difieren solo en la distancia vertical al
  agua, pequeña frente a la altitud), de modo que "aumentar la elevación manteniendo
  constantes las demás variables" no describe una situación real: si el terreno sube
  100 m, ambas variables aumentan 100 m. El modelo reparte el efecto altitudinal casi a
  partes iguales entre las dos, y el efecto interpretable es el **conjunto**. Sumando
  ambos coeficientes por desviación estándar, el efecto es positivo en el extremo alto
  (*krummholz*: 3,76 + 3,66 ≈ 7,4; abeto/pícea: 1,69 + 1,65 ≈ 3,3) y negativo en el
  bajo (álamo/sauce: −2,57 − 2,54 ≈ −5,1; pino ponderosa: −1,80 − 1,73 ≈ −3,5; abeto de
  Douglas: −1,71 − 1,63 ≈ −3,3).
- **La lectura aislada de la razón de *odds* subestima el efecto altitudinal.** La tabla
  anterior compara la razón de *odds* por cada 100 m calculada solo con el coeficiente de
  la elevación con la razón de *odds* conjunta, que combina ambos coeficientes en sus
  escalas originales. Por cada 100 m adicionales de altitud, las *odds* del *krummholz* se
  multiplican por 14,1, y no por 3,8 como sugiere la lectura aislada; las del abeto/pícea,
  por 3,3 en lugar de 1,8. En el extremo bajo, las del álamo/sauce se reducen a 0,16 veces
  su valor (frente a 0,40 en la lectura aislada) y las del pino ponderosa, a 0,28 veces
  (frente a 0,53). Ignorar la colinealidad llevaría, por tanto, a estimar en escala de *log-odds*
  solo la mitad de la intensidad real del gradiente altitudinal.
- **El área silvestre** actúa como ajuste regional. Cache la Poudre tiene coeficientes positivos grandes para el álamo/sauce (3,6), el pino ponderosa y el abeto de Douglas, lo que refleja el proxy geográfico identificado en la auditoría de fuga. Rawah tiene un coeficiente alto para el álamo temblón (3,2), que es más frecuente allí que en las demás áreas.
- **Los índices de sombreado** tienen coeficientes grandes y de signos opuestos para el abeto de Douglas (2,0 a las 9 a. m., −1,7 al mediodía y 2,9 a las 3 p. m.). Es el síntoma típico de la colinealidad detectada en la sección 2.3: los coeficientes se compensan entre sí y no deben interpretarse de forma aislada.
- **Las distancias a carreteras, agua y puntos de incendio** tienen coeficientes pequeños en la mayoría de las clases, con la excepción del álamo/sauce, cuya probabilidad aumenta con la distancia euclidiana al agua (1,4) y con la distancia a carreteras (1,2) cuando las demás variables se mantienen constantes. Como estas variables también están correlacionadas entre sí, estos efectos parciales deben tomarse con cautela.

## 3.11 Revisión crítica del desempeño

El modelo base obtiene un *accuracy* cercano al 72 %, por debajo del rango de 80–90 % a partir del cual un desempeño alto obliga a sospechar de fuga de información o de un problema trivial. Aun así, se revisaron los puntos de control correspondientes:

1. **Fuga de información.** La auditoría de la sección 2.5 no encontró identificadores, variables derivadas del objetivo ni variables posteriores a la predicción. Las predictoras con AUC univariado superior a 0,95 tienen una explicación ecológica directa.
2. **Comparación con la línea base.** La mejora sobre el `DummyClassifier` es sustancial en todas las métricas, incluidas las que dan el mismo peso a todas las clases.
3. **Estructura espacial.** La validación por área silvestre muestra que el desempeño baja para zonas no vistas; el resultado principal se interpreta como desempeño dentro de las áreas muestreadas.
4. **Complejidad del problema.** Para confirmar que el problema no es trivial pero sí tiene margen de mejora, se estima como referencia un modelo no lineal (*gradient boosting* basado en histogramas {cite}`ke2017`) con la misma validación cruzada. No forma parte del modelo base; solo indica cuánta señal queda sin aprovechar.

```{code-cell} ipython3
:tags: [hide-input]

hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
cv_hgb = cross_validate(hgb, X_cv, y_cv, cv=cv5, n_jobs=-1, scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
referencia = pd.DataFrame(
    {
        "Accuracy (validación cruzada)": [resumen_cv.loc["Accuracy", "Validación (media)"], cv_hgb["test_accuracy"].mean()],
        "F1 macro (validación cruzada)": [resumen_cv.loc["F1 macro", "Validación (media)"], cv_hgb["test_f1_macro"].mean()],
    },
    index=["Regresión logística (modelo base)", "Gradient boosting (referencia no lineal)"],
).round(3)
referencia
```

Con los mismos datos y la misma validación, un modelo no lineal sin ajuste de hiperparámetros alcanza 0,81 de *accuracy* y 0,74 de F1 macro, frente a 0,73 y 0,56 de la regresión logística. Esto confirma que el problema **no es trivial para un modelo lineal** y que la señal disponible permite mejoras sustanciales con modelos más flexibles, lo que motiva el benchmark de la Parte II. Al mismo tiempo, cualquier modelo que supere el 90 % de *accuracy* debe someterse de nuevo a la revisión de fuga y a la validación por área; el modelo finalista de la Parte II se somete a esa revisión en la sección 9.4.
