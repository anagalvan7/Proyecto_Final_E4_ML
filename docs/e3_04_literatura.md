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

# 10. Revisión de literatura y evidencia de novedad

La novedad del modelo original debe **demostrarse** con una revisión documentada, no solo declararse. Este
capítulo registra las bases consultadas, las cadenas de búsqueda, la fecha, los
resultados revisados y la matriz de novedad, y termina con una declaración de novedad acotada a lo que la
búsqueda permite afirmar.

```{code-cell} ipython3
:tags: [hide-input]

import glob
import json
from pathlib import Path

import pandas as pd

LIT = Path.cwd().parent / "resultados" / "e3" / "literatura"
archivo = sorted(glob.glob(str(LIT / "*_busquedas.json")))[-1]
registro = json.load(open(archivo, encoding="utf-8"))
cribado = pd.read_csv(LIT / "cribado.csv", encoding="utf-8-sig")
print("Archivo de la búsqueda:", Path(archivo).name)
```

## 10.1 Criterio de uso previo

Se considera uso previo **cualquier trabajo que aplique el modelo a Covertype**, incluidos los artículos de
métodos que usan Covertype como uno de varios conjuntos de prueba, y no solo los dedicados a la cobertura
forestal. Por eso la búsqueda no se limitó a la literatura forestal. También cubrió los artículos de aprendizaje
automático que usan Covertype como referencia, que en este caso son sobre todo los de flujos de datos con
deriva de concepto, donde Covertype es un conjunto clásico.

## 10.2 Fases de la búsqueda

**Fase 1: revisión preliminar en Scopus y Web of Science** (título, resumen y palabras clave). Cruzó tres
conjuntos: clasificación de cobertura vegetal con aprendizaje automático (1.938 artículos, para excluir modelos
ya usados en el tema), clasificación con aprendizaje automático en otros dominios (10.098 artículos, fuente de
candidatos) y Covertype / *forest cover type* con aprendizaje automático (1.539 artículos, verificación final).
Se excluyeron las redes neuronales y las variantes menores de modelos ya usados. De 60 familias candidatas, 4 se
descartaron por aparecer con Covertype (Predictive Clustering Trees, fusión por integral de Choquet, Weighted
kNN y Deep Forest). Las 56 restantes se filtraron por viabilidad de implementación propia y por su potencial
para superar a los modelos del benchmark, lo que dejó tres finalistas: el ensamble por competencia local y por
clase (ECLC), un *boosting* con árboles de hojas logísticas y un fuzzy kNN con regla de pertenencia propia.

**Fase 2: búsqueda documentada y reproducible.** Consultó tres bases bibliográficas con interfaz pública,
guardando la respuesta cruda y la fecha:

- **OpenAlex** {cite}`priem2022`, índice abierto que agrega Crossref, PubMed, arXiv y repositorios
  institucionales. Busca en título, resumen y, cuando está disponible, **texto completo**, lo que permite
  encontrar artículos de métodos que mencionan Covertype solo en su sección experimental.
- **arXiv**, preprints de aprendizaje automático (solo metadatos: título y resumen).
- **Semantic Scholar**, como base complementaria. Su interfaz pública limita la tasa de consultas, por lo que
  solo respondió parte de las cadenas; sus conteos son por relevancia y no se usan para la declaración.

Se usaron **21 cadenas de búsqueda** (11 en la primera ronda y 10 en la segunda). Combinan (i) la familia del modelo o su mecanismo (selección dinámica
de ensambles, competencia local, ponderación dinámica, KNORA, META-DES, pérdida logarítmica local, pesos por
clase), (ii) el problema (Covertype, *forest cover type*, cobertura forestal o vegetal) y (iii) sinónimos en
inglés y en español.

### Segunda ronda: búsqueda ampliada

Como la revisión preliminar solo cubrió títulos, resúmenes y palabras clave, se hizo una segunda ronda en
OpenAlex, que incluye texto completo, y en arXiv (cadenas C12 a C21). Esta ronda cubre las familias de los otros
dos finalistas (árboles con modelos logísticos en las hojas y fuzzy kNN) y la familia del
*stacking*, que el capítulo 11 usa como regla de referencia. Resultados:

- **El *stacking* estándar ya se usó con Covertype**: AutoGluon-Tabular lo usa como conjunto de prueba, y un
  trabajo de 2023 clasifica el tipo de cobertura vegetal con datos cartográficos mediante *stacked
  generalization*. Un modelo que fuera solo *stacking* no sería novedoso en este problema.
- **Los árboles con modelos en las hojas (familia LMT) también tienen uso previo con Covertype**, en
  comparaciones de benchmarks tabulares y en árboles de Hoeffding con perceptrones en las hojas para flujos de
  datos. Esto descarta la novedad del segundo finalista en su familia.
- **Fuzzy kNN no aparece con Covertype**, pero por sí solo su desempeño esperado es bajo frente a los métodos de conjunto.
- El trabajo más cercano a pesos que dependen de la observación es el *Bayesian hierarchical stacking*
  (Yao et al., 2021), sin uso con Covertype; sus pesos dependen de $x$ mediante un modelo bayesiano, no de la
  pérdida local por clase de los vecinos.

Ninguna de las cadenas nuevas encontró el mecanismo del ECLC. Por eso se mantiene como modelo original.

## 10.3 Matriz de novedad

```{code-cell} ipython3
:tags: [hide-input]

filas = []
for r in registro:
    c = cribado[(cribado.id == r["id"]) & (cribado.base == r["base"])]
    filas.append({"ID": r["id"], "Base de datos": r["base"], "Cadena de búsqueda": r["cadena"],
                  "Componentes": r["componentes"], "Fecha": r["fecha"],
                  "Result.": r["resultados"] if r["error"] is None else "error (límite de tasa)",
                  "Revis.": int(c["revisados"].iloc[0]) if len(c) else "—",
                  "Relev.": int(c["relevantes"].iloc[0]) if len(c) else "—",
                  "Observación": c["observacion"].iloc[0] if len(c) else
                      ("Conteo por relevancia; no se usa para la declaración" if r["error"] is None else "Sin respuesta")})
matriz = pd.DataFrame(filas)
matriz = matriz[matriz["Base de datos"] != "DBLP"]
pd.set_option("display.max_colwidth", 400)
matriz
```

La columna «Revis.» indica cuántos resultados se revisaron por título y resumen; «Relev.», cuántos están
relacionados con el mecanismo del ECLC o aplican un método afín a Covertype. En OpenAlex se revisaron
**todos** los resultados de las cadenas con menos de 25 coincidencias y los 25 primeros por relevancia de la
cadena C9 (148 resultados, ninguna sobre el problema). Ninguna cadena devolvió un trabajo que combine
**pesos por clase derivados de la pérdida logarítmica en la vecindad del espacio de variables** con Covertype
ni con la cobertura forestal.

## 10.4 Trabajos más cercanos y diferencias

La comparación se hace **por familia algorítmica**, no solo por aplicación.

| Método afín | Qué hace | Qué conserva el ECLC | Qué es distinto en el ECLC | ¿Usado con Covertype? |
|---|---|---|---|---|
| OLA / LCA {cite}`woods1997` | Elige el clasificador con mayor exactitud en los $k$ vecinos | Competencia medida en una vecindad del espacio de variables | No elige uno: pondera a todos con pesos suaves; mide la competencia con la pérdida logarítmica binaria por clase y no con la exactitud | No encontrado |
| KNORA-E {cite}`ko2008` | Conserva los modelos que aciertan en todos los vecinos | Vecindario de validación con predicciones fuera de muestra | No descarta modelos; pesos en forma cerrada (Proposición 1) | No encontrado |
| META-DES {cite}`cruz2015` | Meta-clasificador que predice si un modelo es competente | Idea de meta-variables de competencia | Sin meta-clasificador entrenado: la competencia es directamente la pérdida, con cota demostrada (Proposición 2) | No encontrado (cadena C7) |
| Competencia probabilística DES-RRC / DES-KL {cite}`woloszynski2011,woloszynski2012` | Competencia probabilística por modelo, ponderada por distancia | Competencia probabilística y pesos por distancia | Pesos distintos para cada clase, contracción hacia la pérdida global y balanceo de los vecinos por clase | No encontrado |
| Selección dinámica para flujos con deriva (ICTAI 2019; CIKM 2025; ADES 2021) | Seleccionan o ponderan modelos según su desempeño reciente en el tiempo | Ponderación dinámica de un ensamble | La competencia depende de la región del espacio de variables, no del tiempo; no hay ventana temporal | **Sí**: Covertype aparece como flujo de prueba (cadenas C1–C3) |
| Ensamble ponderado por clase para flujos {cite}`sarnovsky2021` | Pesos por clase según el desempeño reciente de cada modelo | Pesos por clase | Pesos por clase **locales** en el espacio de variables, derivados de un objetivo regularizado con entropía, con contracción | Posible (resultado de la cadena C5, contexto de flujos) |
| Mayoría ponderada dinámica {cite}`kolter2007` | Pesos multiplicativos que se reducen ante errores en el flujo | Pesos exponenciales | Pesos locales y por clase, sin dimensión temporal | Flujos de datos (no verificado con Covertype) |
| Hedge / pesos exponenciales {cite}`freund1997` | Pesos $\propto e^{-\eta\,\text{pérdida acumulada}}$ globales | Forma exponencial de los pesos y cota $\log M / \beta$ | La pérdida es local (vecinos) y por clase, con contracción | No aplica (algoritmo en línea general) |
| *Stacking* lineal {cite}`wolpert1992` | Combinador lineal global aprendido sobre predicciones fuera de pliegue | Uso de predicciones fuera de pliegue | Pesos que cambian con la región del espacio; sin parámetros entrenados en la capa de combinación | Uso genérico |
| Mezcla de expertos {cite}`jacobs1991` | Red de compuerta que asigna pesos a expertos según $x$ | Pesos que dependen de $x$ | Compuerta no neuronal y no entrenada (forma cerrada a partir de la pérdida local), compatible con la restricción de la guía | — |

**Por qué los métodos existentes no resuelven la limitación abordada.** La limitación del proyecto es doble:
el desempeño relativo de los modelos cambia por región (área silvestre y franja de elevación) y por clase,
con un desbalance de 103:1. Los métodos de selección dinámica clásicos (OLA, KNORA-E, META-DES) miden la
competencia con la exactitud global del modelo en la vecindad, que en un vecindario dominado por las clases
mayoritarias ignora a las clases raras. Los métodos con pesos por clase existentes ponderan por desempeño
reciente en el tiempo, no por región del espacio. El ECLC mide la competencia **para cada clase**, con una
pérdida que penaliza la confianza mal asignada, **en la región de la celda** y con **pesos que compensan el
desbalance**.

## 10.5 Declaración de novedad

```{admonition} Declaración de novedad
:class: important
Al mejor de nuestro conocimiento y según la búsqueda descrita (Scopus y Web of Science; OpenAlex con texto
completo, arXiv y, de forma parcial, Semantic Scholar, con 21 cadenas en inglés y español; consulta del 23 de
octubre de 2026), **no se encontraron estudios que apliquen a Covertype ni a la clasificación de la cobertura
forestal un ensamble cuyos pesos sean, para cada clase, la solución regularizada con entropía de la pérdida
logarítmica binaria en la vecindad de la observación, con contracción hacia la pérdida global y balanceo de
clases en el vecindario.** Los trabajos más cercanos que usan Covertype aplican selección o ponderación dinámica
de ensambles en flujos de datos, con una competencia que depende del tiempo y no de la región del espacio de
variables.
```
