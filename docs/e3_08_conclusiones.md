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

# 13. Conclusiones

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

E3 = Path.cwd().parent / "resultados" / "e3"
v = json.load(open(E3 / "estadistica_veredicto.json", encoding="utf-8"))
vc = json.load(open(E3 / "completo" / "estadistica_veredicto_completo.json", encoding="utf-8"))
print(f"Submuestra de 40.000 celdas: ECLC {v['f1_eclc']:.4f} · mejor benchmark ({v['mejor_benchmark']}) "
      f"{v['f1_mejor_benchmark']:.4f} · superioridad estadística = {v['R3']}")
print(f"Entrenamiento completo: ECLC {vc['f1_eclc']:.4f} (prueba reservada {vc['f1_eclc_prueba_reservada']:.4f}) · "
      f"mejor benchmark ({vc['mejor_benchmark']}) {vc['f1_mejor_benchmark']:.4f} · superioridad estadística = {vc['R3']}")
```

## 13.1 Hallazgos

1. **Las variables cartográficas permiten predecir la cobertura forestal con alta precisión dentro de las áreas
   muestreadas.** Con todo el conjunto de entrenamiento, el modelo original alcanza un F1 macro de 0,915 en la
   prueba reservada, frente a 0,557 de la regresión logística de referencia.
2. **Con el mismo espacio de búsqueda y el mismo presupuesto para las 112 combinaciones del benchmark**, Random
   Forest con SMOTE es la mejor combinación sobre la submuestra de 40.000 celdas (0,783), seguido de K vecinos
   y XGBoost (0,746). El efecto del balanceo depende del algoritmo: mejora a Random Forest y empeora a XGBoost.
3. **El orden de los modelos depende del tamaño del entrenamiento.** Con el conjunto completo, el árbol de
   decisión (0,897) y K vecinos (0,893) superan a Random Forest con SMOTE (0,865) y a XGBoost (0,786), cuyos
   hiperparámetros se eligieron con la submuestra.
4. **El modelo original supera a los siete modelos benchmark.** Con el entrenamiento completo obtiene
   0,907 ± 0,002 en 10 pliegues y 0,915 en la prueba reservada, y gana al mejor benchmark (árbol de decisión) en
   los 10 pliegues ($\delta$ de Cliff = 1). Friedman, Wilcoxon con Holm y la t corregida de Nadeau-Bengio
   confirman la diferencia. Con la submuestra de 40.000 celdas también supera en promedio a los siete
   (0,791 frente a 0,786 de Random Forest; gana en 27 de 30 pliegues), aunque allí la diferencia solo es
   significativa en dos de las tres pruebas.
5. **El modelo original cambia exactitud por equilibrio entre clases.** Mejora sobre todo el F1 de las clases
   raras (álamo/sauce y álamo temblón), a costa de una exactitud global algo menor que la del mejor benchmark.
   Por eso la prueba de McNemar, que compara aciertos, no lo favorece.
6. **El mecanismo funciona fuera de Covertype.** En dígitos manuscritos y cáncer de mama, dos problemas de
   naturaleza distinta, obtiene el mayor F1 macro de los métodos evaluados; en vinos queda por debajo de Random
   Forest.
7. **Hay combinadores más potentes.** El *stacking* lineal y HistGradientBoosting obtienen un F1 macro mayor que
   el modelo original en ambos tamaños de entrenamiento. Con el entrenamiento completo, reglas más simples (pesos
   por error global y KNORA-E) también lo superan, y no todos sus componentes aportan. El modelo original mejora
   a los modelos que combina, pero no es el mejor clasificador posible para este problema.
8. **La elevación es el determinante principal de la cobertura** en todos los modelos analizados, seguida del
   tipo de suelo, el área silvestre y las distancias a carreteras y a puntos de incendio.

## 13.2 Novedad, implementación y superioridad del modelo original

| Criterio | Resultado | Evidencia |
|---|---|---|
| **Novedad** | No se encontró el mecanismo del ECLC (pesos por clase en forma cerrada sobre la pérdida logarítmica binaria local, con contracción y balanceo de vecinos) aplicado a Covertype ni a la cobertura forestal | Capítulo 10 |
| **Implementación propia** | Clase compatible con scikit-learn; la búsqueda de vecinos, las pérdidas, la contracción, la regla de pesos y la combinación están programadas con numpy. La regla se deriva de un objetivo explícito y tiene propiedades demostradas | Capítulo 11 |
| **Superioridad estadística** | Con el entrenamiento completo: sí (tres pruebas frente al mejor benchmark, efecto grande). Con la submuestra de 40.000 celdas: dos de tres pruebas | Capítulo 12 |

## 13.3 Limitaciones

- **Magnitud de la mejora.** La ventaja sobre el mejor benchmark es de alrededor de una centésima de F1 macro con
  el entrenamiento completo y de media centésima con la submuestra.
- **Hiperparámetros de los benchmarks.** Se eligieron con la submuestra de 40.000 celdas y un espacio de búsqueda
  acotado, y se mantuvieron fijos con el entrenamiento completo para todos los modelos por igual. Una búsqueda más
  amplia podría mejorar a cualquiera de ellos.
- **Dependencia entre pliegues.** Los pliegues de la validación repetida comparten datos de entrenamiento. La t
  corregida de Nadeau-Bengio lo compensa, pero las pruebas de rangos pueden ser algo optimistas.
- **Generalización geográfica.** Los modelos están validados dentro de las cuatro áreas silvestres muestreadas,
  no en áreas nuevas.
- **Cobertura de la revisión de literatura.** El cribado se hizo por título y resumen, con texto completo solo
  donde OpenAlex lo indexa.

## 13.4 Trabajo futuro

1. Usar los pesos locales del modelo original como entrada de un combinador lineal regularizado, para capturar
   las correcciones entre clases que hoy logra el *stacking*.
2. Repetir el benchmark con un espacio de búsqueda más amplio, común a todos los algoritmos, y reoptimizar los
   hiperparámetros con el conjunto completo.
3. Evaluar el modelo con validación por área silvestre y en escenarios de flujo de datos.
4. Usar un índice de vecinos aproximados para reducir el costo de inferencia con conjuntos de referencia grandes.

## 13.5 Declaración de autoría y uso de herramientas

Ana Galván, Valeria Oñate y María José Ruiz son las autoras de este trabajo. El análisis se realizó con
scikit-learn, imbalanced-learn, XGBoost, Optuna, DEAP, SHAP, LIME, statsmodels, scikit-posthocs y Jupyter Book.
Se utilizó Claude (Anthropic) como herramienta de asistencia en la programación de los experimentos y en la
redacción del documento.
