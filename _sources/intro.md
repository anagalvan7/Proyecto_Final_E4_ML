# Clasificación de la cobertura forestal a partir de variables cartográficas

**Curso:** Machine Learning  
**Entrega:** Entregable 4 (final)  
**Autoras:** Ana Galván, Valeria Oñate y María José Ruiz  
**Fecha:** 23 de octubre de 2026

## Resumen

La cartografía de la cobertura forestal es un insumo básico para el manejo de bosques, la evaluación del
riesgo de incendio y la conservación de hábitats, pero su levantamiento en campo es costoso y, en áreas
protegidas o remotas, difícil de mantener actualizado. Este estudio evalúa en qué medida variables
cartográficas (topografía, hidrografía, red vial, área administrativa y tipo de suelo) permiten predecir el
tipo de cobertura dominante en celdas de 30 × 30 m del Roosevelt National Forest (Colorado, Estados Unidos).
Usa el conjunto *Covertype*: 581.012 celdas, 54 predictoras y 7 clases con una razón de desbalance de 103:1
{cite}`uci_covertype,blackard2000`.

La **Parte I** documenta el análisis exploratorio, una auditoría de fuga de información, el preprocesamiento
y un modelo base de regresión logística multinomial (F1 macro de 0,558 en la prueba reservada). La
**Parte II** construye un diseño factorial de **112 combinaciones** (7 algoritmos × 4 estrategias de
balanceo × 4 métodos de optimización), con el mismo espacio de búsqueda y el mismo presupuesto para todas,
evaluadas con validación cruzada anidada. **Random Forest con SMOTE es la mejor combinación** (F1 macro de
0,783 en la validación anidada y de 0,866 en la prueba reservada con el conjunto completo).

La **Parte III** propone un modelo original, el **ensamble por competencia local y por clase (ECLC)**. Para
cada celda nueva y cada cobertura, pondera a los siete modelos del benchmark según su pérdida logarítmica en
las celdas vecinas, con pesos en forma cerrada derivados de un objetivo regularizado con entropía, contracción
hacia la pérdida global y balanceo de clases. Se demuestran la unicidad de los pesos, una cota de la pérdida
local y la complejidad del algoritmo.

**Con el conjunto de entrenamiento completo, el ECLC alcanza un F1 macro de 0,907 en validación cruzada y de
0,915 en la prueba reservada.** Supera al mejor benchmark (árbol de decisión, 0,897) en los 10 pliegues, con
diferencias significativas según las pruebas de Friedman, Wilcoxon con corrección de Holm y t corregida de
Nadeau-Bengio. Con la submuestra de 40.000 celdas también supera en promedio a los siete benchmarks, aunque la
diferencia con el mejor es menor. El *stacking* lineal y HistGradientBoosting obtienen un F1 macro mayor que el
ECLC.

**Palabras clave:** clasificación multiclase, cobertura forestal, desbalance de clases, validación cruzada
anidada, selección dinámica de ensambles, competencia local, pruebas de hipótesis.

## Pregunta de investigación

¿Puede un ensamble que pondera a cada modelo según su desempeño local y por clase superar, de forma
estadísticamente significativa, a los mejores modelos del benchmark al predecir la cobertura forestal a partir
de variables cartográficas? Y, como base para responderla, ¿qué combinación de algoritmo, tratamiento del
desbalance y estrategia de optimización ofrece el mejor desempeño cuando todas se comparan en igualdad de
condiciones?

## Objetivos

1. Comparar 112 combinaciones de algoritmo, balanceo y optimización con el mismo espacio de búsqueda y el mismo
   presupuesto, y seleccionar con criterios estadísticos la mejor combinación de cada algoritmo (Parte II).
2. Diseñar e implementar un modelo original con novedad documentada y propiedades demostradas (capítulos 10 y 11).
3. Validar estadísticamente su superioridad frente a los modelos del benchmark (capítulo 12).

## Organización del documento

| Parte | Capítulo | Contenido |
|---|---|---|
| I | 1–4. Datos, EDA y modelo base | Base de datos, análisis exploratorio, fuga y estructura, preprocesamiento, modelo base |
| II | 5. Benchmark combinatorio | 112 combinaciones con validación anidada, finalista y evaluación en la prueba reservada |
| II | 6. Optimización de hiperparámetros | Grid, Random, Bayesiana (TPE) y Genética; desempeño en cualquier momento y *Successive Halving* |
| II | 7. Comparación estadística | Friedman, Nemenyi, DeLong, *bootstrap* y tamaño del efecto |
| II | 8. Interpretabilidad | SHAP global y local, y comparación con LIME |
| II | 9. Eficiencia y conclusiones del benchmark | Complejidad, implementaciones optimizadas, paralelización, revisión crítica |
| III | 10. Revisión de literatura | Bases, cadenas de búsqueda, matriz de novedad, trabajos cercanos y declaración de novedad |
| III | 11. Modelo original (ECLC) | Formulación, demostraciones, algoritmo, complejidad, pruebas, ablación, robustez y SHAP |
| III | 12. Validación estadística | Protocolo, supuestos, pruebas con Holm, intervalos, tamaños del efecto y entrenamiento completo |
| III | 13. Conclusiones | Hallazgos, limitaciones, trabajo futuro y declaración de autoría |
| III | 14. Reproducibilidad | Entorno, semillas, orden de ejecución y tabla maestra de experimentos |
| — | Apéndices | Código auxiliar, registro de cambios del benchmark y bibliografía |
