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

# 12. Validación estadística del modelo original

Este capítulo responde a la pregunta central: **¿supera el ECLC a los modelos benchmark de forma
estadísticamente significativa y no trivial?** La métrica es el F1 macro, la misma del benchmark (sección 5.1).

```{code-cell} ipython3
:tags: [hide-input]

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image, display

RAIZ = Path.cwd().parent
sys.path.insert(0, str(RAIZ / "scripts"))
E3 = RAIZ / "resultados" / "e3"
resumen = pd.read_csv(E3 / "estadistica_resumen.csv", encoding="utf-8-sig")
pareadas = pd.read_csv(E3 / "estadistica_pareadas.csv", encoding="utf-8-sig")
veredicto = json.load(open(E3 / "estadistica_veredicto.json", encoding="utf-8"))
holdout = {r["modelo"]: r for r in json.load(open(E3 / "holdout_resumen.json", encoding="utf-8"))}
import glob
filas = []
for f in sorted(glob.glob(str(E3 / "meta" / "rep*_f?.json"))):
    d = json.load(open(f, encoding="utf-8"))
    for k, v in d["metricas"].items():
        filas.append({"rep": d["rep"], "pliegue": d["pliegue"], "metodo": k, "f1": v["f1_macro"]})
piv = pd.DataFrame(filas).pivot_table(index=["rep", "pliegue"], columns="metodo", values="f1")
NOMBRE = {"ECLC": "ECLC (original)", "base:KNN": "KNN", "base:NB": "Naive Bayes", "base:LR": "Reg. logística",
          "base:DT": "Árbol", "base:RF": "Random Forest", "base:XGB": "XGBoost", "base:SVM": "SVM",
          "HGB (estado del arte)": "HistGradientBoosting"}
```

## 12.1 Protocolo

- **Pareado estricto.** El ECLC y los 7 modelos benchmark (la mejor combinación de cada uno, sección 5.4) se
  evalúan sobre **las mismas particiones externas**, con la misma semilla (42), la misma submuestra
  estratificada de 40.000 celdas y el mismo preprocesamiento. Los hiperparámetros de cada modelo base son los
  que eligió su bucle interno en el benchmark.
- **Validación cruzada externa repetida.** Con solo 3 pliegues, una prueba de Wilcoxon no puede alcanzar
  $p < 0{,}125$ y la potencia es nula. Por eso se usan **10 repeticiones × 3 pliegues = 30 pliegues
  pareados**. La repetición 0 es exactamente la partición del benchmark (`StratifiedKFold(3, shuffle=True,
  random_state=42)`); la repetición $r$ usa la semilla $42 + r$. En la repetición 0, cada modelo base usa
  los hiperparámetros que su bucle interno eligió en ese pliegue; en las demás, la configuración más frecuente
  de sus tres pliegues.
- **Presupuesto equivalente.** Los hiperparámetros propios del ECLC $(k,\beta,\tau)$ se optimizan **dentro**
  de cada pliegue externo, con optimización bayesiana (TPE) de 6 ensayos y validación interna de 3 pliegues:
  el mismo presupuesto del optimizador bayesiano del benchmark.
- **Prueba reservada.** Los 7 modelos y el ECLC se reentrenaron con las 40.000 celdas y se evaluaron **una
  sola vez** en las 116.203 celdas de la prueba reservada (sección 1.6). Esa evaluación alimenta McNemar.
- **Competidor adicional del estado del arte.** HistGradientBoosting, el *boosting* por histogramas de
  scikit-learn inspirado en LightGBM (Ke et al., 2017), se optimizó con Random Search de 6 configuraciones en
  las mismas particiones. No forma parte de los 7 modelos benchmark, pero se reporta por honestidad.

```{code-cell} ipython3
:tags: [hide-input]

r = resumen.copy()
r["metodo"] = r["metodo"].map(lambda m: NOMBRE.get(m, m))
r = r.rename(columns={"metodo": "Modelo / regla", "f1_media": "F1 macro (media 30 pliegues)", "f1_sd": "DE",
                      "ic95_bca_inf": "IC 95 % BCa inf.", "ic95_bca_sup": "IC 95 % BCa sup.",
                      "accuracy_media": "Accuracy", "tiempo_entrenamiento_s": "Entrenamiento (s)",
                      "tiempo_inferencia_s": "Inferencia (s)"})
r.round(4)
```

La tabla resume los 30 pliegues para todos los modelos y reglas de combinación: media ± desviación estándar
del F1 macro, intervalo de confianza bootstrap BCa de la media y tiempos medios de entrenamiento e inferencia
por pliegue. El tiempo de entrenamiento del ECLC incluye las predicciones fuera de pliegue de los 7 modelos,
su reentrenamiento y la optimización de $(k,\beta,\tau)$.

```{code-cell} ipython3
:tags: [hide-input]

pr = pd.DataFrame([{"Modelo": h.get("nombre", m), **{k: round(v, 4) for k, v in h["prueba_reservada"].items()
                    if k != "f1_por_clase"}} for m, h in holdout.items()]).sort_values("f1_macro", ascending=False)
pr
```

La segunda tabla reporta las mismas métricas en la prueba reservada (116.203 celdas). Es una sola
evaluación por modelo, sin variabilidad entre pliegues; su función es confirmar el orden observado en la
validación cruzada con datos que ningún modelo vio.

## 12.2 Supuestos de cada prueba

```{code-cell} ipython3
:tags: [hide-input]

mejor = "base:RF"
dif = piv["ECLC"] - piv[mejor]
sw = stats.shapiro(dif)
asim = stats.skew(dif)
print(f"Diferencias ECLC − Random Forest en 30 pliegues: media = {dif.mean():.4f}, DE = {dif.std(ddof=1):.4f}")
print(f"Shapiro-Wilk sobre las diferencias: W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}; asimetría = {asim:.2f}")
```

| Prueba | Supuestos | Verificación en este estudio |
|---|---|---|
| Friedman + Nemenyi | Bloques independientes; rangos comparables entre tratamientos | Los 30 pliegues son bloques; los pliegues de una misma repetición son disjuntos y las repeticiones usan semillas distintas. La dependencia por solapamiento de entrenamiento entre repeticiones hace que el p-valor sea algo optimista; Nemenyi es conservador y lo compensa en parte |
| Wilcoxon de rangos con signo | Diferencias pareadas, distribución simétrica alrededor de la mediana | Pareado estricto; la asimetría de las diferencias se reporta arriba |
| t corregida de Nadeau-Bengio | Normalidad aproximada de las diferencias; corrige la varianza por el solapamiento de los conjuntos de entrenamiento ($n_{prueba}/n_{entrenamiento} = 0{,}5$) | Shapiro-Wilk arriba; la corrección es la recomendada por la guía para validación repetida |
| McNemar | Observaciones independientes, mismo conjunto de prueba para ambos modelos | Una evaluación en la prueba reservada (celdas distintas). La autocorrelación espacial (sección 2.7) viola en parte la independencia, por eso no es la única prueba |
| Bayesiana de rangos con signo | Diferencias pareadas; ROPE definida a priori | ROPE = ±0,01 en F1 macro, fijada antes de ver los resultados (una centésima es la mínima diferencia que el equipo consideró relevante) |

Los datos son de corte transversal (no hay orden temporal), así que se usan pruebas para observaciones
independientes; Diebold-Mariano, Giacomini-White y el MCS, pensadas para series de tiempo, no aplican.

## 12.3 Prueba global: Friedman y diagrama de diferencia crítica

```{code-cell} ipython3
:tags: [hide-input]

f = veredicto["friedman"]
print(f"Friedman: χ² = {f['estadistico']:.2f}, k = {f['k']} modelos, N = {f['N']} pliegues, p = {f['p_valor']:.2e}")
print(f"Diferencia crítica de Nemenyi (α = 0,05): CD = {f['cd']:.3f}")
display(pd.Series(f["rangos"], name="rango promedio").sort_values().to_frame())
display(Image(filename=str(E3 / "diagrama_cd_eclc.png")))
```

El universo de la prueba incluye el ECLC, los 7 benchmarks y HistGradientBoosting. Friedman rechaza que todos
tengan el mismo desempeño. El diagrama ubica a cada modelo por su rango promedio en los 30 pliegues. Las barras
azules unen grupos que Nemenyi no distingue, porque su diferencia de rango es menor que la CD.

## 12.4 Comparaciones dirigidas con corrección de Holm

Se comparó el ECLC con cada uno de los 8 competidores (7 benchmarks + HistGradientBoosting) con tres pruebas
frecuentistas unilaterales ($H_0$: el ECLC no es mejor). Por cada prueba se hacen **8 comparaciones**, y sus
p-valores se ajustan con Holm-Bonferroni. En total, la familia de pruebas pareadas suma $3 \times 8 = 24$
contrastes frecuentistas, más la prueba ómnibus y la bayesiana.

```{code-cell} ipython3
:tags: [hide-input]

t = pareadas[["competidor", "dif_media_f1", "dif_ic95_inf", "dif_ic95_sup", "wilcoxon_W", "wilcoxon_p",
              "wilcoxon_p_holm", "nb_t", "nb_p", "nb_p_holm", "mcnemar_b01", "mcnemar_b10", "mcnemar_p",
              "mcnemar_p_holm"]].copy()
t.columns = ["Competidor", "Δ F1", "IC95 inf", "IC95 sup", "W", "p Wilcoxon", "p Wilcoxon (Holm)", "t NB",
             "p NB", "p NB (Holm)", "ECLC acierta / comp. falla", "ECLC falla / comp. acierta", "p McNemar",
             "p McNemar (Holm)"]
t.style.format({c: "{:.2e}" for c in t.columns if c.startswith("p ")} | {"Δ F1": "{:.4f}", "IC95 inf": "{:.4f}",
               "IC95 sup": "{:.4f}", "W": "{:.0f}", "t NB": "{:.2f}"})
```

### Tabla de pruebas de hipótesis frente al mejor benchmark

```{code-cell} ipython3
:tags: [hide-input]

fila = pareadas[pareadas.competidor == veredicto["mejor_benchmark"]].iloc[0]
tabla = pd.DataFrame([
    {"Prueba": "Friedman (ómnibus, 9 modelos)", "H0": "Todos los modelos tienen el mismo rango esperado",
     "Supuestos verificados": "Bloques = pliegues pareados", "Estadístico": f"χ² = {f['estadistico']:.2f}",
     "p crudo": f["p_valor"], "p ajustado": f["p_valor"], "Efecto": "—",
     "Conclusión": "Rechaza" if f["p_valor"] < 0.05 else "No rechaza"},
    {"Prueba": "Nemenyi (post hoc)", "H0": "Rango ECLC = rango del mejor benchmark",
     "Supuestos verificados": "Friedman significativo", "Estadístico":
         f"Δ rango = {f['rangos'][veredicto['mejor_benchmark']] - f['rangos']['ECLC (original)']:.2f} (CD = {f['cd']:.2f})",
     "p crudo": np.nan, "p ajustado": np.nan, "Efecto": "—",
     "Conclusión": "Rechaza" if veredicto["pruebas_vs_mejor"]["Nemenyi (rango ECLC vs mejor benchmark > CD)"] else "No rechaza"},
    {"Prueba": "Wilcoxon de rangos con signo", "H0": "Mediana de (ECLC − RF) ≤ 0", "Supuestos verificados":
         "Pareado; simetría (12.2)", "Estadístico": f"W = {fila['wilcoxon_W']:.0f}", "p crudo": fila["wilcoxon_p"],
     "p ajustado": fila["wilcoxon_p_holm"], "Efecto": f"δ Cliff = {fila['cliff_delta']:.2f} ({fila['magnitud_cliff']})",
     "Conclusión": "Rechaza" if fila["wilcoxon_p_holm"] < 0.05 else "No rechaza"},
    {"Prueba": "t corregida de Nadeau-Bengio", "H0": "E[ECLC − RF] ≤ 0", "Supuestos verificados":
         "Normalidad aprox. (Shapiro 12.2); varianza corregida", "Estadístico": f"t = {fila['nb_t']:.2f}",
     "p crudo": fila["nb_p"], "p ajustado": fila["nb_p_holm"], "Efecto": f"d de Cohen (pareada) = {fila['cohen_dz']:.2f}",
     "Conclusión": "Rechaza" if fila["nb_p_holm"] < 0.05 else "No rechaza"},
    {"Prueba": "McNemar (prueba reservada)", "H0": "P(ECLC acierta, RF falla) ≤ P(ECLC falla, RF acierta)",
     "Supuestos verificados": "Mismo conjunto de prueba; independencia aproximada",
     "Estadístico": f"χ² = {fila['mcnemar_chi2']:.1f} (b01 = {fila['mcnemar_b01']}, b10 = {fila['mcnemar_b10']})",
     "p crudo": fila["mcnemar_p"], "p ajustado": fila["mcnemar_p_holm"], "Efecto": "—",
     "Conclusión": "Rechaza" if fila["mcnemar_p_holm"] < 0.05 else "No rechaza"},
    {"Prueba": "Bayesiana de rangos con signo", "H0": "—  (ROPE = ±0,01)", "Supuestos verificados": "Pareado",
     "Estadístico": f"P(ECLC mejor) = {fila['bayes_p_eclc_mejor']:.3f}; P(ROPE) = {fila['bayes_p_rope']:.3f}",
     "p crudo": np.nan, "p ajustado": np.nan, "Efecto": "—",
     "Conclusión": "ECLC mejor" if fila["bayes_p_eclc_mejor"] > 0.95 else
                   "Equivalencia práctica" if fila["bayes_p_rope"] > 0.95 else "Indeterminada"},
])
tabla.style.format({"p crudo": "{:.2e}", "p ajustado": "{:.2e}"}, na_rep="—")
```

## 12.5 Tamaño del efecto e intervalos

```{code-cell} ipython3
:tags: [hide-input]

e = pareadas[["competidor", "dif_media_f1", "dif_ic95_inf", "dif_ic95_sup", "cliff_delta", "magnitud_cliff",
              "cohen_dz", "ecl_gana_pliegues", "bayes_p_eclc_mejor", "bayes_p_rope", "bayes_p_competidor_mejor"]]
e.columns = ["Competidor", "Δ F1 medio", "IC95 BCa inf", "IC95 BCa sup", "δ Cliff", "Magnitud", "d Cohen (dz)",
             "Pliegues que gana el ECLC (de 30)", "P(ECLC mejor)", "P(ROPE)", "P(competidor mejor)"]
e.round(4)
```

El delta de Cliff compara las distribuciones de F1 de los 30 pliegues, con los umbrales de Romano et al.
(2006). Mide qué tanto se separan las dos distribuciones, mientras que Wilcoxon solo indica si el signo de las
diferencias pareadas es consistente. Por eso, una diferencia pequeña en valor absoluto pero estable entre
pliegues puede ser significativa con un $\delta$ moderado.

## 12.6 Evaluación en conjuntos de datos adicionales

```{code-cell} ipython3
:tags: [hide-input]

ad = pd.read_csv(E3 / "datasets_adicionales.csv", encoding="utf-8-sig")
res = ad.groupby(["conjunto", "metodo"])["f1_macro"].agg(["mean", "std"]).round(4).unstack(0)
display(res)
filas = []
for cj, g in ad.groupby("conjunto"):
    p = g.pivot_table(index=["rep", "pliegue"], columns="metodo", values="f1_macro")
    bases = [c for c in p.columns if c.startswith("base:")]
    mejor = p[bases].mean().idxmax()
    d = p["ECLC"] - p[mejor]
    w = stats.wilcoxon(d, alternative="greater", zero_method="zsplit") if np.any(d != 0) else None
    filas.append({"Conjunto": cj, "Mejor base": mejor.replace("base:", ""), "F1 mejor base": p[mejor].mean(),
                  "F1 ECLC": p["ECLC"].mean(), "F1 stacking": p["stacking lineal"].mean(),
                  "F1 HGB": p["HGB (estado del arte)"].mean(), "Δ ECLC − mejor base": d.mean(),
                  "p Wilcoxon (unilateral)": w.pvalue if w else 1.0})
pd.DataFrame(filas).round(4)
```

Los tres conjuntos son de naturaleza distinta a Covertype: dígitos manuscritos (imágenes de 8 × 8 píxeles,
10 clases), análisis químico de vinos (3 clases, 178 muestras) y cáncer de mama (variables clínicas, 2
clases). Se usaron las mismas 7 familias, los mismos espacios de búsqueda y el mismo presupuesto (Random Search
de 6 configuraciones para cada modelo base y para HistGradientBoosting, 6 ensayos de TPE para el ECLC), con 5
repeticiones × 3 pliegues. Con 15 pliegues y conjuntos pequeños donde varios modelos ya superan 0,95 de F1,
las diferencias son del orden de una centésima y la potencia es baja. Estos resultados indican si el ECLC
**generaliza a otros problemas sin degradarse**; no buscan demostrar superioridad en ellos.

## 12.7 Veredicto de superioridad

```{code-cell} ipython3
:tags: [hide-input]

v = veredicto
print(f"Mejor benchmark: {v['mejor_benchmark']} (F1 macro {v['f1_mejor_benchmark']:.4f}); ECLC: {v['f1_eclc']:.4f}")
print(f"1) Media del ECLC mayor que la de los 7 benchmarks: {v['condicion_1_media_mejor_que_los_7']}")
for k, ok in v["pruebas_vs_mejor"].items():
    print(f"   - {k}: {'rechaza H0' if ok else 'no rechaza H0'}")
print(f"2) Al menos tres pruebas (incluida una global) rechazan H0 con p ajustado < 0,05: {v['condicion_2_tres_pruebas']}")
print(f"3) Tamaño del efecto no despreciable (|δ Cliff| ≥ 0,147): {v['condicion_3_efecto_no_despreciable']} "
      f"(δ = {v['cliff_vs_mejor']:.3f})")
print(f"\nSuperioridad estadística = {v['R3']}")
```

Se exige que se cumplan tres condiciones simultáneas para declarar la superioridad:

1. La media del ECLC debe superar a la mejor combinación benchmark y a la mejor de cada uno de los 7 modelos.
2. Al menos tres pruebas distintas deben rechazar la igualdad con p ajustado < 0,05, una de ellas global.
3. El efecto no debe ser despreciable.

La celda anterior evalúa las tres con los resultados obtenidos. El capítulo 13 discute el resultado. Dos
hechos deben leerse junto con el veredicto, sea cual sea:

- **La superioridad se mide frente a los 7 modelos benchmark, no frente a cualquier método.**
  HistGradientBoosting y el *stacking* lineal obtienen un F1 macro mayor que el ECLC en los mismos 30 pliegues
  (tabla de la sección 12.1). El ECLC no es el mejor clasificador posible para Covertype; es una regla de
  combinación que mejora de forma consistente a los modelos que combina.
- **La diferencia frente a Random Forest es pequeña en valor absoluto** (alrededor de media centésima de F1
  macro). Es estable entre pliegues, pero su relevancia práctica para un inventario forestal es limitada.

## 12.8 Protocolo de datos completos

Las secciones anteriores comparan los modelos entrenados con unas 27.000 celdas (pliegues de la submuestra de
40.000 del benchmark). En Covertype el desempeño depende mucho del tamaño del entrenamiento: con todo el
conjunto de entrenamiento, K vecinos y el árbol de decisión superan 0,9 de F1 macro. Por eso la comparación se
repitió **con las 464.809 celdas de entrenamiento y el mismo protocolo para todos los modelos**:

- **Partición:** validación cruzada externa 5 × 2 (`StratifiedKFold(5, shuffle=True, random_state=42 + r)`,
  r = 0, 1), es decir, 10 pliegues pareados con unas 371.800 celdas de entrenamiento y 93.000 de prueba cada uno,
  más la prueba reservada de 116.203 celdas entrenando con las 464.809.
- **Modelos benchmark:** la mejor combinación de cada uno de los 7 algoritmos, con el balanceo y la
  configuración elegidos en el benchmark de 40.000 celdas. No se reoptimizaron: los hiperparámetros quedan
  fijos para todos por igual. El SVM sigue limitado a 6.000 celdas de entrenamiento, como en su benchmark.
- **Modelo original:** el mismo ECLC, sin cambios, con $(k, \beta, \tau)$ optimizados por TPE con 6 ensayos en
  validación interna de cada pliegue.
- **Competidor del estado del arte:** HistGradientBoosting con la configuración más frecuente de su búsqueda en
  los 30 pliegues de 40.000 celdas.

```{code-cell} ipython3
:tags: [hide-input]

C = E3 / "completo"
rc = pd.read_csv(C / "estadistica_resumen_completo.csv", encoding="utf-8-sig")
rc["metodo"] = rc["metodo"].map(lambda m: NOMBRE.get(m, m))
rc.columns = ["Modelo / regla", "F1 macro (media 10 pliegues)", "DE", "IC 95 % BCa inf.", "IC 95 % BCa sup.",
              "Accuracy", "F1 prueba reservada"]
rc.round(4)
```

La tabla resume los 10 pliegues y la prueba reservada para los benchmarks, el ECLC, sus variantes de ablación
y las reglas de combinación estándar.

```{code-cell} ipython3
:tags: [hide-input]

vc = json.load(open(C / "estadistica_veredicto_completo.json", encoding="utf-8"))
pc = pd.read_csv(C / "estadistica_pareadas_completo.csv", encoding="utf-8-sig")
display(pc[["competidor", "dif_media_f1", "dif_ic95_inf", "dif_ic95_sup", "ecl_gana_pliegues", "wilcoxon_p_holm",
            "nb_p_holm", "mcnemar_p_holm", "bayes_p_eclc_mejor", "cliff_delta", "magnitud_cliff"]].round(4))
display(Image(filename=str(C / "diagrama_cd_completo.png")))
print(f"Mejor benchmark: {vc['mejor_benchmark']} ({vc['f1_mejor_benchmark']:.4f}); ECLC: {vc['f1_eclc']:.4f} "
      f"(prueba reservada: {vc['f1_eclc_prueba_reservada']:.4f})")
for k_, ok in vc["pruebas_vs_mejor"].items():
    print(f"   - {k_}: {'rechaza H0' if ok else 'no rechaza H0'}")
print(f"Superioridad estadística con datos completos = {vc['R3']}")
```

Con 10 pliegues, la prueba de Wilcoxon solo puede alcanzar un p-valor unilateral de $1/1024$ si el ECLC gana en
todos. Tras ajustar por Holm las 8 comparaciones, ese mínimo queda en $0{,}008$. La t corregida usa
$n_{prueba}/n_{entrenamiento} = 0{,}25$.

### Lectura de los resultados con datos completos

- **El ECLC supera 0,9 de F1 macro y a los 7 benchmarks.** En los 10 pliegues obtiene 0,907 ± 0,002 (0,915 en
  la prueba reservada) frente a 0,897 del árbol de decisión, el mejor benchmark con datos completos, y gana en
  los 10 pliegues ($\delta$ de Cliff = 1). Friedman, Wilcoxon-Holm y la t corregida de Nadeau-Bengio rechazan
  la igualdad frente al árbol, así que **la superioridad estadística se cumple en este protocolo**. Nemenyi no separa
  a los dos modelos (diferencia de rangos menor que la CD). McNemar no favorece al ECLC: el árbol tiene mayor
  exactitud y el ECLC mayor F1 macro, el mismo intercambio entre clases mayoritarias y raras de la sección 11.8.
- **El orden de los benchmarks cambia con el tamaño de muestra.** Con 27.000 celdas lideraba Random Forest con
  SMOTE; con 371.000, el árbol de decisión sin límite de profundidad y K vecinos superan 0,89, mientras que
  Random Forest (limitado a profundidad 20) y XGBoost (profundidad 6) quedan atrás, porque sus hiperparámetros
  se eligieron con 40.000 celdas y no se reoptimizaron.
- **Resultados desfavorables.** El *stacking* lineal (0,932) y HistGradientBoosting (0,935) superan al ECLC
  también con datos completos. Además, con datos completos la ablación ya no respalda todos los componentes:
  las variantes sin pesos por clase, sin contracción y sin núcleo de distancia obtienen un F1 macro igual o
  ligeramente mayor que el ECLC completo. Los pesos por error global (0,921) y KNORA-E (0,912) también lo
  superan. La contribución del ECLC en este protocolo se limita a superar a los modelos que combina; no supera
  a reglas de combinación más simples.
- **Los hiperparámetros $(k, \beta, \tau)$ elegidos coinciden en los 11 casos** ($k = 9$, $\beta = 0{,}66$,
  $\tau = 42$). Con la misma semilla, el TPE propone los mismos 6 ensayos, y el mejor resulta igual en todas
  las particiones.
