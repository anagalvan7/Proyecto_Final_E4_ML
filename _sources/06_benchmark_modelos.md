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

# 5. Benchmark combinatorio de modelos

La Parte I estableció que la regresión logística multinomial alcanza un F1 macro de 0,558
en la prueba reservada y que su limitación es de sesgo, no de varianza (sección 3.9). Esta
parte del documento aborda la pregunta que se deriva de ese diagnóstico: **¿qué familia de
modelos, qué estrategia de balanceo de clases y qué método de optimización de
hiperparámetros permiten superar ese desempeño de forma estadística y prácticamente
significativa?**

Para responderla se construye un diseño factorial completo de **7 algoritmos × 4
estrategias de balanceo × 4 métodos de optimización = 112 combinaciones**. Cada
combinación se entrena y se evalúa de forma independiente mediante validación cruzada
anidada, y todas comparten la partición, la semilla y el preprocesamiento de la Parte I.
Como *Covertype* no contiene una variable respuesta continua, el estudio se restringe a la
rama de clasificación; la regresión no tiene un objetivo natural en este conjunto de datos.

```{code-cell} ipython3
:tags: [hide-input]

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent / "scripts"))
import _utils3 as u3

sns.set_theme(style="whitegrid", context="notebook")
RESULTS3 = Path.cwd().parent / "resultados"
tabla = pd.read_csv(RESULTS3 / "tabla_maestra.csv", encoding="utf-8-sig")
tabla["f1_macro_pliegues"] = tabla["f1_macro_pliegues"].apply(
    lambda s: json.loads(s) if isinstance(s, str) else s
)
print(f"Combinaciones registradas: {len(tabla)} de 112")
print(f"Aplicables: {int(tabla['aplicable'].sum())} · "
      f"No aplicables (balanceo no soportado por el algoritmo): {int((~tabla['aplicable']).sum())}")
```

## 5.1 Diseño experimental

### Métrica principal de validación

El criterio de selección en todas las etapas es el **F1 macro**, es decir, la media no
ponderada del F1 de las siete clases. La elección responde a la estructura del problema:
con una razón de desbalance de 103:1 (sección 1.7), la *accuracy* está dominada por las
dos clases mayoritarias, que suman el 85 % de las celdas, y puede mejorar aunque el modelo
ignore por completo las clases raras. Esto fue lo que ocurrió con la regresión logística,
que alcanzó 0,72 de *accuracy* con un *recall* de 0,04 en el álamo temblón (sección 3.5).
El F1 macro asigna el mismo peso a cada clase y penaliza tanto los falsos positivos como
los falsos negativos, lo que lo hace adecuado para un inventario forestal en el que omitir
un ecosistema raro y sobrestimarlo son errores de costo comparable. La *accuracy*, la
exactitud balanceada, el AUC uno-contra-el-resto y el *log loss* se reportan como métricas
complementarias.

### Validación cruzada anidada

Seleccionar hiperparámetros y estimar el desempeño con los mismos pliegues produce una
estimación optimista, conocida como sesgo de selección del modelo {cite}`cawley2010,varma2006`. Para evitarlo, cada combinación se
evalúa con dos bucles:

- **Bucle externo (3 pliegues estratificados):** estima el desempeño de generalización.
  Cada pliegue externo deja fuera un tercio de la submuestra, que no interviene en la
  búsqueda de hiperparámetros.
- **Bucle interno (3 pliegues estratificados):** dentro de cada partición de entrenamiento
  externa, ejecuta el método de optimización completo y elige la configuración con mayor
  F1 macro medio.

Todas las métricas que se reportan en los capítulos 5 a 7 provienen del bucle externo.
El preprocesamiento (ingeniería de variables, escalado y, cuando corresponde, SMOTE o
ADASYN) se encapsula en un `Pipeline` de scikit-learn o de imbalanced-learn y se ajusta
dentro de cada pliegue, de modo que ninguna observación de validación interviene en el
ajuste de las transformaciones ni en la generación de muestras sintéticas.

**Número de pliegues.** Con tres pliegues externos e internos, cada evaluación interna se
entrena con unas 17.800 celdas y se valida con unas 8.900. En consecuencia, la clase menos
frecuente (álamo/sauce, 0,47 %) aporta alrededor de 40 observaciones a cada partición de
validación interna, suficientes para estimar su F1. Con cinco o diez pliegues, el número
de ajustes se multiplicaría por un factor de 2,8 a 11 sin una ganancia proporcional en la
precisión de la estimación, porque el error de muestreo del F1 macro entre pliegues ya es
del orden de 0,005 (sección 5.4). La contrapartida es que tres pliegues externos ofrecen
poca potencia a las pruebas no paramétricas sobre rangos, un aspecto que se analiza en el
capítulo 7.

### Submuestra para el benchmark

Entrenar 112 combinaciones con validación anidada sobre las 464.809 filas de
entrenamiento implicaría decenas de miles de ajustes de modelos de conjunto. La curva de
aprendizaje de la sección 3.9 muestra que el F1 macro de validación del modelo lineal se
estabiliza a partir de unas 30.000 celdas; con base en ese resultado, el benchmark se
ejecuta sobre una **submuestra estratificada de 40.000 filas** del conjunto de
entrenamiento (`N_BENCHMARK` en `_utils3.py`). La SVM con núcleo RBF, cuyo costo de
entrenamiento crece entre $O(n^2)$ y $O(n^3)$, se evalúa sobre una submuestra estratificada
de 9.000 filas.

```{note}
La submuestra modifica únicamente el número de filas de **entrenamiento** empleadas en la
búsqueda y en la validación anidada. La partición 80/20, la semilla y el preprocesamiento
son los de la Parte I. El modelo finalista se reentrena con el conjunto de entrenamiento
completo antes de evaluarse, una sola vez, en la prueba reservada (sección 5.5).
```

## 5.2 Algoritmos y espacios de búsqueda

Los rangos se fijaron a partir de la escala del problema (57 predictoras tras la
ingeniería de variables, 40.000 filas y 7 clases) y de los resultados de la Parte I, no de
valores arbitrarios. Los parámetros continuos de regularización y suavizado se exploran en
escala logarítmica.

| Algoritmo | Hiperparámetros explorados | Justificación del rango |
|---|---|---|
| KNN {cite}`cover1967` | `n_neighbors` ∈ {5, 11, 21, 41}; `weights` ∈ {uniform, distance} | Valores impares, desde vecindades locales hasta suavizados amplios; la ponderación por distancia favorece la frontera local en las zonas de solapamiento entre clases (sección 2.3) |
| Naive Bayes gaussiano {cite}`hand2001` | `var_smoothing` ∈ [10⁻¹⁰, 10⁻⁶] (log) | Intervalo estándar de suavizado de la varianza, que estabiliza las variables binarias de suelo con varianza casi nula |
| Regresión logística (L1/L2) {cite}`hosmer2013` | `C` ∈ [0,1; 10] (log); `penalty` ∈ {L1, L2} | Cubre C = 10, el óptimo de la sección 3.3, y C = 1, con un F1 de validación prácticamente idéntico; la penalización L1 permite contrastar la selección implícita de variables frente a la contracción L2 |
| Árbol de decisión {cite}`breiman1984` | `max_depth` ∈ {5, 10, 20, sin límite}; `min_samples_leaf` ∈ {1, 10, 50} | Desde árboles interpretables hasta árboles completamente desarrollados, con hojas mínimas que controlan el sobreajuste |
| Random Forest | `n_estimators` ∈ {100, 200}; `max_depth` ∈ {10, 20} | Profundidad máxima de 20 para acotar el costo de cada ajuste en la validación anidada; el mismo espacio se usa con las cuatro estrategias de balanceo |
| XGBoost (`hist`) | `n_estimators` ∈ {100, 150}; `max_depth` ∈ {4, 6}; `learning_rate` ∈ [0,05; 0,1] | Árboles poco profundos y tasas de aprendizaje moderadas, el rango habitual del *boosting*; el mismo espacio se usa con las cuatro estrategias de balanceo |
| SVM (núcleo RBF) {cite}`cortes1995` | `C` ∈ [0,1; 10] (log); `gamma` ∈ {scale, 0,01, 0,1} | `gamma = "scale"` es la referencia de scikit-learn; los otros dos valores exploran fronteras más y menos suaves |

El presupuesto de cada método por pliegue externo fue el siguiente: Grid Search recorre la
rejilla completa (entre 4 y 12 configuraciones, según el algoritmo, idénticas para todas sus variantes de balanceo); Random Search evalúa 6
configuraciones; la optimización bayesiana, 6 ensayos de Optuna, y el algoritmo genético,
una población de 4 individuos durante 2 generaciones. Estos presupuestos son deliberadamente
pequeños frente a los habituales en la literatura, una restricción impuesta por el costo de
la validación anidada sobre 112 combinaciones en una sola estación de trabajo. Sus
consecuencias se discuten en los capítulos 6 y 9.

## 5.3 Estrategias de balanceo

- **Sin balanceo:** línea base, idéntica al modelo de la Parte I.
- **SMOTE** {cite}`chawla2002`: genera observaciones sintéticas de las clases minoritarias
  por interpolación entre vecinos cercanos, en el espacio de las 57 variables ya
  transformadas.
- **ADASYN** {cite}`he2008`: variante adaptativa de SMOTE que genera más observaciones en
  las regiones donde la clase minoritaria es más difícil de aprender, es decir, donde
  predominan los vecinos de otras clases.
- **Ponderación por clase:** `class_weight="balanced"` en los algoritmos que lo admiten
  (regresión logística, árbol, Random Forest y SVM). En XGBoost y Naive Bayes se usa el
  equivalente funcional, `sample_weight`, con pesos inversamente proporcionales a la
  frecuencia de cada clase en el pliegue de entrenamiento. KNN no admite ninguno de los dos
  mecanismos; esas cuatro combinaciones se declaran no aplicables en lugar de forzar una
  aproximación artificial.

```{code-cell} ipython3
:tags: [hide-input]

no_aplicables = tabla.loc[~tabla["aplicable"], ["modelo_nombre", "balanceo_nombre", "optimizacion_nombre"]]
no_aplicables.columns = ["Algoritmo", "Balanceo", "Optimización"]
display(no_aplicables.style.hide(axis="index"))
```

## 5.4 Resultados del benchmark

La tabla siguiente resume, para cada par algoritmo–balanceo, el F1 macro medio del bucle
externo, promediado sobre los cuatro métodos de optimización, junto con el mínimo y el
máximo alcanzados.

```{code-cell} ipython3
:tags: [hide-input]

aplic = tabla[tabla["aplicable"]].copy()
resumen = (aplic.groupby(["modelo_nombre", "balanceo_nombre"])
           .agg(media=("f1_macro_media", "mean"), minimo=("f1_macro_media", "min"),
                maximo=("f1_macro_media", "max"))
           .round(3).sort_values("media", ascending=False))
resumen.index.names = ["Algoritmo", "Balanceo"]
resumen.columns = ["F1 macro medio", "Mínimo", "Máximo"]
display(resumen)
```

```{code-cell} ipython3
:tags: [hide-input]

top15 = aplic.sort_values("f1_macro_media", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
etiquetas = top15["modelo_nombre"] + " · " + top15["balanceo_nombre"] + " · " + top15["optimizacion_nombre"]
ax.barh(etiquetas, top15["f1_macro_media"], xerr=top15["f1_macro_sd"], color="#2d6a4f", capsize=2)
ax.set_xlabel("F1 macro (media ± desviación estándar entre pliegues externos)")
ax.set_title("Las 15 mejores combinaciones de las 108 aplicables")
ax.set_xlim(0.6, 0.85)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

mejor = aplic.loc[aplic["f1_macro_media"].idxmax()]
base_lr = aplic[(aplic["modelo"] == "LR") & (aplic["balanceo"] == "ninguno")]["f1_macro_media"].max()
print(f"Mejor combinación: {mejor['modelo_nombre']} · {mejor['balanceo_nombre']} · {mejor['optimizacion_nombre']}")
print(f"F1 macro (bucle externo) = {mejor['f1_macro_media']:.4f} ± {mejor['f1_macro_sd']:.4f}")
print(f"Regresión logística sin balanceo (misma submuestra): {base_lr:.4f}")
print(f"Diferencia absoluta: {mejor['f1_macro_media'] - base_lr:+.4f}")
```

Los resultados, obtenidos con el mismo espacio de búsqueda y el mismo presupuesto para las 112
combinaciones, permiten cuatro observaciones:

1. **Los métodos de conjunto basados en árboles y los vecinos dominan la clasificación.** Random Forest
   {cite}`breiman2001` con SMOTE (0,783) ocupa el primer lugar, seguido de K vecinos (0,746), XGBoost
   {cite}`chen2016` (0,746) y el árbol de decisión individual (0,722). La regresión logística (0,564) y Naive
   Bayes (0,224) quedan al final. Esto confirma que las fronteras entre clases son no lineales y que el
   supuesto de independencia condicional de Naive Bayes no es compatible con la fuerte dependencia entre
   elevación, área silvestre y tipo de suelo (sección 2.3).
2. **El efecto del balanceo depende del algoritmo.** En Random Forest, SMOTE y ADASYN elevan el F1 macro en
   0,033 y los pesos de clase en 0,025 frente a no balancear. En XGBoost, las tres estrategias lo reducen entre
   0,023 y 0,040. El sobremuestreo ayuda a un bosque de árboles profundos a definir regiones para las clases
   raras, mientras que en el *boosting* desplaza las fronteras de las clases mayoritarias.
3. **La variabilidad entre pliegues es pequeña.** La desviación estándar del F1 macro entre pliegues externos se
   mantiene alrededor de 0,01 o por debajo en las combinaciones mejor clasificadas, de modo que las diferencias
   de varias centésimas entre familias de algoritmos superan con amplitud la variabilidad intrínseca de cada
   combinación.
4. **Varias combinaciones empatan exactamente.** Por ejemplo, Random Forest con SMOTE alcanza el mismo F1
   macro con Grid, Random y Bayesiano, porque en un espacio discreto de cuatro configuraciones los tres métodos
   convergen a la misma solución. Este fenómeno condiciona la comparación entre optimizadores del capítulo 6.

La diferencia entre el mejor modelo y la regresión logística es una estimación puntual. Su significancia
estadística y su tamaño de efecto se evalúan en el capítulo 7, y con 30 pliegues en el capítulo 12.

## 5.5 Modelo finalista reentrenado con el conjunto completo

El finalista del benchmark es **Random Forest con SMOTE** (`n_estimators = 200`,
`max_depth = 20`, la configuración más frecuente entre los tres pliegues externos). Se reentrena con las
**464.809 filas de entrenamiento** (con SMOTE, cerca de 1,6 millones de filas tras el sobremuestreo) y se
evalúa **una sola vez** sobre las 116.203 filas de la prueba reservada en la sección 1.6.

```{code-cell} ipython3
:tags: [hide-input]

with open(RESULTS3 / "resumen_finalista.json", encoding="utf-8") as f:
    resumen_final = json.load(f)

print(f"Finalista: {resumen_final['finalista']} · hiperparámetros: {resumen_final['hiperparametros']}")
print(f"F1 macro en la prueba reservada (finalista, entrenamiento completo): {resumen_final['f1_macro_finalista']:.4f}")
print(f"F1 macro en la prueba reservada (regresión logística): {resumen_final['f1_macro_referencia']:.4f}")
print(f"Diferencia absoluta: {resumen_final['ganancia_absoluta']:+.4f}")
```

El F1 macro sube de 0,783 en la submuestra a **0,866** con el conjunto completo. Con 11,6 veces más datos de
entrenamiento, el álamo/sauce pasa de unas 190 observaciones en la submuestra a unas 2.200, y el álamo
temblón, de unas 650 a unas 7.600. A diferencia de la regresión logística, cuya curva de aprendizaje se aplana
a partir de 30.000 celdas (sección 3.9), un bosque de árboles profundos aprovecha esa información adicional.
Por la misma razón, la estimación del bucle externo debe interpretarse como un **límite inferior** del
desempeño alcanzable con el conjunto completo.

```{code-cell} ipython3
:tags: [hide-input]

delong = pd.read_csv(RESULTS3 / "delong_finalistas.csv", encoding="utf-8-sig")
delong["clase"] = delong["clase"].map(u3.CLASS_SHORT)
delong.round(4)
```

La prueba de DeLong {cite}`delong1988` compara, clase por clase (uno contra el resto), el AUC del finalista
reentrenado con el conjunto completo con el de la regresión logística de referencia. La diferencia es
significativa en las siete clases tras la corrección de Holm. Es mayor en las dos clases mayoritarias
(abeto/pícea y pino lodgepole), que el modelo lineal apenas separaba entre sí.

## 5.6 Evaluación de los modelos en la prueba reservada

Para la comparación con el modelo original (capítulos 11 y 12), los siete modelos se reentrenaron con la
submuestra de 40.000 celdas del benchmark, con la mejor combinación de cada uno, de modo que todos se comparan
con los mismos datos. La tabla resume su desempeño en la prueba reservada.

```{code-cell} ipython3
:tags: [hide-input]

E3 = RESULTS3 / "e3"
hold = {r["modelo"]: r for r in json.load(open(E3 / "holdout_resumen.json", encoding="utf-8"))}
tabla_h = pd.DataFrame([{"Modelo": r["nombre"], "Balanceo": r.get("balanceo"), "Optimización": r.get("optimizacion"),
                         "F1 macro": r["prueba_reservada"]["f1_macro"], "Accuracy": r["prueba_reservada"]["accuracy"],
                         "Exactitud balanceada": r["prueba_reservada"]["exactitud_balanceada"],
                         "Log loss": r["prueba_reservada"]["log_loss"]}
                        for m, r in hold.items() if m not in ("ECLC", "HGB")]).sort_values("F1 macro", ascending=False)
tabla_h.round(3)
```

```{code-cell} ipython3
:tags: [hide-input]

h = np.load(E3 / "holdout_probas.npz")
y_test = h["y_test"]
P_rf = h["RF"] / h["RF"].sum(1, keepdims=True)
pred_rf = P_rf.argmax(1) + 1
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
cm = confusion_matrix(y_test, pred_rf, labels=u3.CLASES, normalize="true")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1,
            xticklabels=[u3.CLASS_SHORT[c] for c in u3.CLASES], yticklabels=[u3.CLASS_SHORT[c] for c in u3.CLASES])
ax.set(xlabel="Clase predicha", ylabel="Clase real",
       title="Random Forest + SMOTE (40.000 celdas): matriz de confusión normalizada por fila")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for i, c in enumerate(u3.CLASES):
    fpr, tpr, _ = roc_curve(y_test == c, P_rf[:, i])
    pre, rec, _ = precision_recall_curve(y_test == c, P_rf[:, i])
    ax[0].plot(fpr, tpr, label=u3.CLASS_SHORT[c], color=u3.CLASS_PALETTE[c])
    ax[1].plot(rec, pre, label=u3.CLASS_SHORT[c], color=u3.CLASS_PALETTE[c])
ax[0].plot([0, 1], [0, 1], "k:", alpha=0.5)
ax[0].set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curvas ROC uno-contra-el-resto")
ax[1].set(xlabel="Recall", ylabel="Precisión", title="Curvas de precisión-recall uno-contra-el-resto")
ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.show()
recall = pd.Series(np.diag(cm), index=[u3.CLASS_SHORT[c] for c in u3.CLASES])
recall.to_frame("Recall de Random Forest").round(3).T
```

Los errores siguen el patrón ecológico identificado en la sección 3.7. La confusión dominante se produce entre
el abeto/pícea y el pino lodgepole, las dos clases mayoritarias, que comparten la franja subalpina. La segunda
fuente de error es el álamo temblón, que se confunde con el pino lodgepole en el piso montano superior. El
balanceo con SMOTE sube el *recall* de las clases raras a costa de algo de precisión.

## 5.7 Calibración

Un modelo puede discriminar bien entre clases sin que sus probabilidades reflejen la frecuencia real de
ocurrencia. La calibración se evalúa con el **Brier score multiclase** (error cuadrático medio entre el vector
de probabilidades predichas y la codificación *one-hot* de la clase real) y con el **error de calibración
esperado** {cite}`naeini2015` (ECE, la discrepancia media ponderada entre confianza y frecuencia observada en 10
intervalos de igual frecuencia), por clase.

```{code-cell} ipython3
:tags: [hide-input]

def ece_clase(y_bin, p, bins=10):
    orden = np.argsort(p)
    grupos = np.array_split(orden, bins)
    return sum(len(g) * abs(p[g].mean() - y_bin[g].mean()) for g in grupos) / len(p)


Y = np.eye(7)[y_test - 1]
filas = []
for m in ["RF", "XGB", "KNN", "LR"]:
    P = h[m] / h[m].sum(1, keepdims=True)
    filas.append({"Modelo": hold[m]["nombre"], "Brier multiclase": ((P - Y) ** 2).sum(1).mean(),
                  **{f"ECE {u3.CLASS_SHORT[c]}": ece_clase(Y[:, i], P[:, i]) for i, c in enumerate(u3.CLASES)}})
display(pd.DataFrame(filas).set_index("Modelo").round(4))

fig, ax = plt.subplots(figsize=(6.5, 6))
for i, c in enumerate(u3.CLASES):
    orden = np.argsort(P_rf[:, i])
    grupos = np.array_split(orden, 10)
    ax.plot([P_rf[g, i].mean() for g in grupos], [Y[g, i].mean() for g in grupos], "o-",
            color=u3.CLASS_PALETTE[c], label=u3.CLASS_SHORT[c], alpha=0.8)
ax.plot([0, 1], [0, 1], "k:", label="Calibración perfecta")
ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada",
       title="Random Forest + SMOTE: diagrama de confiabilidad por clase")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
```

El Brier multiclase de Random Forest (0,238), XGBoost (0,255) y K vecinos (0,228) es mucho menor que el de la
regresión logística (0,382): además de clasificar mejor, asignan probabilidades más fiables. En Random Forest,
el error de calibración se concentra en las **dos clases mayoritarias** (ECE de 0,079 en el abeto/pícea y de
0,101 en el pino lodgepole), mientras que las cinco clases minoritarias quedan bien calibradas (ECE ≤ 0,03).
Es la distorsión esperada de SMOTE: al entrenar con proporciones de clase artificialmente iguales, el modelo
**subestima** la probabilidad de las clases grandes, y XGBoost sin balanceo, entrenado con las proporciones
reales, las calibra mejor (ECE de 0,037 y 0,035). Como el criterio del proyecto es el F1 macro, que depende
del *argmax* y no de la escala de las probabilidades, no se recalibra. El modelo original del capítulo 11 usa
la pérdida logarítmica de estas probabilidades para ponderar a los modelos, y las propiedades de su regla no
dependen de una calibración perfecta.
