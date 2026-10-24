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

# 7. Comparación estadística de los modelos

```{admonition} El problema de escala
:class: warning
Con 108 combinaciones aplicables, el número de comparaciones por pares posibles es
$\binom{108}{2} = 5.778$. Aplicar una prueba pareada a cada una con $\alpha = 0{,}05$ y sin
corrección produciría, en promedio, unos 289 falsos positivos atribuibles al azar. La
comparación sigue por ello una **estrategia jerárquica** {cite}`demsar2006`: una prueba
ómnibus (Friedman) determina si existe alguna diferencia; solo si resulta significativa se
aplica un análisis *post hoc* (Nemenyi) y, para un subconjunto reducido de modelos, una
comparación dirigida (DeLong) con corrección por comparaciones múltiples y tamaño del
efecto.
```

```{code-cell} ipython3
:tags: [hide-input]

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

sys.path.insert(0, str(Path.cwd().parent / "scripts"))
import _utils3 as u3

sns.set_theme(style="whitegrid", context="notebook")
RESULTS3 = Path.cwd().parent / "resultados"

tabla = pd.read_csv(RESULTS3 / "tabla_maestra.csv", encoding="utf-8-sig")
tabla["f1_macro_pliegues"] = tabla["f1_macro_pliegues"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
aplic = tabla[tabla["aplicable"]].copy()
ranking = pd.read_csv(RESULTS3 / "friedman_ranking.csv", encoding="utf-8-sig")
with open(RESULTS3 / "friedman_resumen.json", encoding="utf-8") as f:
    resumen = json.load(f)
```

## 7.1 Etapa 1: prueba ómnibus de Friedman sobre las 108 combinaciones

La prueba de Friedman {cite}`friedman1937` es el análogo no paramétrico del ANOVA de medidas
repetidas. Ordena las combinaciones dentro de cada pliegue externo (los bloques) según su
F1 macro y contrasta la hipótesis nula de que todas tienen el mismo rango promedio.

```{code-cell} ipython3
:tags: [hide-input]

print(f"Combinaciones (k): {resumen['n_combinaciones']} · Pliegues externos (N): {resumen['n_pliegues']}")
print(f"Estadístico de Friedman: χ² = {resumen['estadistico_friedman']:.2f} con {resumen['grados_libertad']} g.l.")
print(f"Valor p: {resumen['p_valor']:.2e}")
```

El estadístico ($\chi^2_F = 316{,}7$; 107 g.l.; $p = 1{,}3 \times 10^{-22}$) rechaza la
hipótesis nula: al menos una combinación difiere de las demás en su rango promedio.

## 7.2 Etapa 2: análisis *post hoc* de Nemenyi

La prueba de Nemenyi {cite}`nemenyi1963` compara todas las combinaciones controlando el
error de tipo I de forma conjunta. Dos combinaciones se consideran estadísticamente
distintas cuando sus rangos promedio difieren en más que la **diferencia crítica**:

$$CD = q_\alpha \sqrt{\frac{k(k+1)}{6N}}.$$

```{code-cell} ipython3
:tags: [hide-input]

cd_108 = resumen["diferencia_critica_nemenyi"]
rango_max = resumen["n_combinaciones"] - 1
print(f"Diferencia crítica con k = 108 y N = 3: CD = {cd_108:.1f}")
print(f"Máxima diferencia posible entre rangos promedio: {rango_max}")
```

Con $k = 108$ y solo $N = 3$ pliegues, la diferencia crítica (110,9) **supera la máxima
diferencia posible entre rangos** (107). En consecuencia, la prueba de Nemenyi no puede
declarar distinto ningún par de combinaciones, ni siquiera la mejor frente a la peor. No se
trata de evidencia de equivalencia, sino de una limitación de potencia: la prueba fue
diseñada para comparar algoritmos sobre muchos conjuntos de datos {cite}`demsar2006`, y su
diferencia crítica decrece con $\sqrt{N}$. Con tres bloques, la comparación de 108
combinaciones carece de resolución.

Para obtener conclusiones con la información disponible, el análisis se repite al nivel en
que se toman las decisiones de modelado: se selecciona **la mejor combinación de cada
algoritmo** (siete familias) y se aplica de nuevo la secuencia Friedman–Nemenyi.

```{code-cell} ipython3
:tags: [hide-input]

mejores = aplic.loc[aplic.groupby("modelo")["f1_macro_media"].idxmax()].copy()
mejores = mejores.sort_values("f1_macro_media", ascending=False)
M = np.array(mejores["f1_macro_pliegues"].tolist())          # familias × pliegues
chi2, p_fam = stats.friedmanchisquare(*M)
rangos = np.array([stats.rankdata(-M[:, j]) for j in range(M.shape[1])]).T
k, N = M.shape
q_alpha = 2.949                                              # Nemenyi, α = 0,05, k = 7
cd_7 = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

familias = pd.DataFrame({
    "Algoritmo": mejores["modelo_nombre"].values,
    "Mejor combinación": (mejores["balanceo_nombre"] + " · " + mejores["optimizacion_nombre"]).values,
    "F1 macro (media)": mejores["f1_macro_media"].values,
    "DE entre pliegues": mejores["f1_macro_sd"].values,
    "Rango promedio": rangos.mean(axis=1),
}).set_index("Algoritmo").round(4)
display(familias)
print(f"Friedman (7 familias, N = 3): χ² = {chi2:.2f} con {k - 1} g.l., p = {p_fam:.4f}")
print(f"Diferencia crítica de Nemenyi: CD = {cd_7:.2f}")
```

```{code-cell} ipython3
:tags: [hide-input]

nem = sp.posthoc_nemenyi_friedman(M.T)
nombres = list(familias.index)
nem.index = pd.Index(nombres)
nem.columns = pd.Index(nombres, name="Valor p de Nemenyi")
display(nem.round(3))
```

```{code-cell} ipython3
:tags: [hide-input]

# Diagrama de diferencia crítica (formato de Demšar, 2006): eje de rangos promedio (1 = mejor,
# a la izquierda); las barras rojas unen grupos de algoritmos que no difieren
# significativamente según la CD de Nemenyi.
orden = familias["Rango promedio"].sort_values()
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0.4, 7.6)
ax.set_ylim(0.85 - 0.22 - 0.11 * 3 - 0.08, 1.2)
y_eje = 0.85
ax.plot([1, 7], [y_eje, y_eje], color="black", lw=1)
for r in range(1, 8):
    ax.plot([r, r], [y_eje, y_eje + 0.03], color="black", lw=1)
    ax.text(r, y_eje + 0.06, str(r), ha="center", va="bottom", fontsize=9)
nombres = list(orden.index)
mitad = (len(nombres) + 1) // 2
for i, nombre in enumerate(nombres):
    r = orden[nombre]
    if i < mitad:
        y = y_eje - 0.22 - 0.11 * i
        ax.plot([r, r, 0.95], [y_eje, y, y], color="#1b4965", lw=0.9)
        ax.text(0.9, y, f"{nombre} ({r:.1f})", ha="right", va="center", fontsize=8.5)
    else:
        y = y_eje - 0.22 - 0.11 * (len(nombres) - 1 - i)
        ax.plot([r, r, 7.05], [y_eje, y, y], color="#1b4965", lw=0.9)
        ax.text(7.1, y, f"{nombre} ({r:.1f})", ha="left", va="center", fontsize=8.5)
# Grupos no distinguibles: secuencias maximales de algoritmos cuyo rango difiere < CD.
rs = orden.values
grupos = []
for a in range(len(rs)):
    b = max(j for j in range(len(rs)) if rs[j] - rs[a] < cd_7)
    if b > a and not any(a >= g0 and b <= g1 for g0, g1 in grupos):
        grupos.append((a, b))
for k_g, (a, b) in enumerate(grupos):
    yb = y_eje - 0.07 - 0.05 * k_g
    ax.plot([rs[a] - 0.05, rs[b] + 0.05], [yb, yb], color="#c0392b", lw=3, solid_capstyle="round")
ax.plot([1, 1 + cd_7], [1.08, 1.08], color="black", lw=1.8)
ax.plot([1, 1], [1.05, 1.11], color="black", lw=1.8)
ax.plot([1 + cd_7, 1 + cd_7], [1.05, 1.11], color="black", lw=1.8)
ax.text(1 + cd_7 / 2, 1.12, f"CD = {cd_7:.2f}", ha="center", va="bottom", fontsize=9)
ax.axis("off")
plt.tight_layout()
plt.show()
```

Random Forest ocupa el primer lugar en los tres pliegues externos. K vecinos y XGBoost se alternan en el
segundo y el tercer lugar (sus F1 medios difieren en 0,0006), y el resto del orden se mantiene. El estadístico de
Friedman ($\chi^2_F = 17{,}7$; 6 g.l.; $p = 0{,}007$) rechaza la igualdad de rangos. Aun así, con $N = 3$ la
diferencia crítica de Nemenyi es de 5,20 rangos, y solo el par **Random Forest–Naive Bayes** (diferencia de 6
rangos) resulta significativo ($p = 0{,}012$). Random Forest frente a la regresión logística ($p = 0{,}069$)
queda en el límite. El orden de las familias es estable, pero tres bloques no bastan para que un procedimiento
de comparación múltiple tan conservador como el de Nemenyi lo convierta en significancia. Por esta razón, el
capítulo 12 repite la comparación con **30 pliegues pareados**.

### Diferencias entre modelos frente a variabilidad intrínseca

La desviación estándar del F1 macro entre pliegues externos está entre 0,004 y 0,011 en
las cuatro mejores familias. Las diferencias entre familias consecutivas son de un orden de
magnitud mayor: 0,037 entre Random Forest y KNN, 0,024 entre XGBoost y el árbol de decisión y 0,219
entre Random Forest y la regresión logística. La excepción es el par KNN–XGBoost (0,0006), que no se distingue. Las diferencias entre familias superan
claramente la variabilidad intrínseca de cada una, lo que motiva la comparación dirigida
con el conjunto de prueba completo que se presenta a continuación, donde el tamaño de
muestra sí otorga potencia.

## 7.3 Comparación dirigida: prueba de DeLong

La prueba de DeLong {cite}`delong1988` compara las áreas bajo dos curvas ROC correlacionadas,
obtenidas sobre las mismas observaciones, mediante la varianza asintótica de los
estadísticos U de Mann-Whitney {cite}`mann1947`. Se aplica al subconjunto reducido que interesa para la
conclusión del estudio: el modelo finalista frente a la regresión logística de la
Parte I, ambos evaluados sobre las 116.203 celdas de la prueba reservada. Al tratarse de
un problema multiclase, la comparación se realiza por clase (uno contra el resto), y los
siete valores $p$ se ajustan con el método de Holm {cite}`holm1979`.

```{code-cell} ipython3
:tags: [hide-input]

delong = pd.read_csv(RESULTS3 / "delong_finalistas.csv", encoding="utf-8-sig")
delong["clase"] = delong["clase"].map(u3.CLASS_SHORT)
delong = delong.rename(columns={"clase": "Clase", "auc_finalista": "AUC finalista",
                                "auc_referencia": "AUC regresión logística", "delta_auc": "Δ AUC",
                                "z": "z", "p_valor": "p crudo", "p_holm": "p ajustado (Holm)"})
display(delong.style.hide(axis="index").format({"AUC finalista": "{:.4f}", "AUC regresión logística": "{:.4f}",
                                               "Δ AUC": "{:+.4f}", "z": "{:.1f}",
                                               "p crudo": lambda v: "< 1e-300" if v == 0 else f"{v:.1e}",
                                               "p ajustado (Holm)": lambda v: "< 1e-300" if v == 0 else f"{v:.1e}"}))
```

Las siete comparaciones favorecen al finalista (Random Forest con SMOTE, reentrenado con el conjunto
completo) con valores $p$ ajustados por debajo de la precisión de la máquina ($z$ entre 12 y 131). La
magnitud de la mejora, sin embargo, es muy desigual entre clases. El AUC aumenta alrededor de 0,12 y 0,13 en el
abeto/pícea y el pino lodgepole, y en 0,08 en el álamo temblón. En cambio, en el álamo/sauce, una clase que el
modelo lineal ya discriminaba con un AUC de 0,997, la mejora es de apenas 0,002. Con 116.203 observaciones
incluso esa diferencia es estadísticamente significativa, lo que ilustra por qué el tamaño del efecto debe
acompañar siempre al valor $p$.

## 7.4 Intervalos de confianza por *bootstrap*

```{code-cell} ipython3
:tags: [hide-input]

from sklearn.metrics import f1_score, accuracy_score
h = np.load(RESULTS3 / "e3" / "holdout_probas.npz")
y_t = h["y_test"]
rng = np.random.default_rng(42)
filas_ic = []
for m, nombre in [("RF", "Random Forest + SMOTE"), ("LR", "Regresión logística")]:
    pred = h[m].argmax(1) + 1
    idx = [rng.integers(0, len(y_t), len(y_t)) for _ in range(200)]
    f1b = np.array([f1_score(y_t[i], pred[i], average="macro") for i in idx])
    accb = np.array([accuracy_score(y_t[i], pred[i]) for i in idx])
    for metrica, b, v in [("F1 macro", f1b, f1_score(y_t, pred, average="macro")),
                          ("Accuracy", accb, accuracy_score(y_t, pred))]:
        filas_ic.append({"Modelo": nombre, "Métrica": metrica, "Valor": v,
                         "IC 95 % inferior": np.percentile(b, 2.5), "IC 95 % superior": np.percentile(b, 97.5)})
display(pd.DataFrame(filas_ic).set_index(["Modelo", "Métrica"]).round(4))
```

Los intervalos se obtuvieron con el método de percentiles del *bootstrap* {cite}`efron1993`, con 200 remuestreos
con reemplazo de las 116.203 celdas de prueba, para los dos modelos reentrenados con la submuestra de 40.000
celdas (sección 5.6). Con un tamaño de muestra de este orden, la distribución *bootstrap* del F1 macro es
prácticamente simétrica, por lo que la corrección BCa {cite}`efron1987` produciría límites indistinguibles a la
precisión reportada. Los intervalos de Random Forest y de la regresión logística están separados por más de 0,2
unidades de F1 macro. Los intervalos BCa sobre los 30 pliegues de la validación repetida están en el capítulo 12.

## 7.5 Tamaño del efecto

La significancia estadística no implica relevancia práctica, especialmente con más de cien
mil observaciones. Como medida de tamaño del efecto se usa el **delta de Cliff** {cite}`cliff1993`, una
medida no paramétrica de dominancia estocástica coherente con el análisis basado en rangos:
$\delta = P(X > Y) - P(X < Y)$, con valores entre −1 y 1. Se calcula sobre los F1 macro
de los pliegues externos del benchmark, que son evaluaciones sobre datos distintos. No se
calcula sobre las réplicas *bootstrap* de la prueba reservada: esas réplicas remuestrean
un único conjunto de prueba, no son observaciones independientes y su dispersión refleja
solo el error de muestreo de ese conjunto, de modo que cualquier diferencia sistemática
produciría $|\delta| = 1$ y el tamaño del efecto quedaría inflado. El *bootstrap* se
reserva para los intervalos de confianza de la sección 7.4.

```{code-cell} ipython3
:tags: [hide-input]

def cliff_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sign(a[:, None] - b[None, :])).mean()

pl = dict(zip(mejores["modelo_nombre"], mejores["f1_macro_pliegues"]))
filas = []
for otro in familias.index[1:]:
    filas.append({"Comparación": f"{familias.index[0]} frente a {otro}", "Nivel": "Pliegues externos (3 frente a 3)",
                  "δ de Cliff": cliff_delta(pl[familias.index[0]], pl[otro])})
display(pd.DataFrame(filas).set_index("Comparación").round(3))
```

En los pliegues externos, Random Forest domina por completo a las otras seis familias ($\delta = 1$ en todos
los casos). Según los umbrales de {cite:t}`romano2006` ($|\delta| \geq 0{,}474$), se trata de un efecto grande.
Con tres pliegues por modelo, sin embargo, el delta se calcula sobre solo nueve pares, por lo que es una
estimación gruesa: indica una dominancia consistente, no su magnitud exacta. Este resultado complementa al de
Nemenyi: aunque tres pliegues no bastan para la significancia formal de un procedimiento de comparaciones
múltiples, la dominancia observada es completa y consistente.

## 7.6 Síntesis

- La prueba de Friedman confirma diferencias entre las 108 combinaciones, pero la prueba de Nemenyi carece de
  potencia con tres pliegues externos: su diferencia crítica excede el rango posible.
- Al nivel de las siete familias, Random Forest ocupa el primer lugar en los tres pliegues y domina a las demás
  ($\delta = 1$), aunque Nemenyi solo declara significativa su diferencia con Naive Bayes. K vecinos y
  XGBoost no se distinguen entre sí.
- En la prueba reservada, donde el tamaño de muestra sí otorga potencia, el finalista supera a la regresión
  logística en las siete clases (DeLong con Holm), con intervalos de confianza claramente separados.
- Para ganar potencia, la comparación del modelo original del capítulo 12 usa una validación cruzada repetida
  10 × 3 (30 pliegues pareados).
