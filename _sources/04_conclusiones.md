# 4. Conclusiones de la Parte I

## 4.1 Hallazgos principales

**Sobre la base de datos.** El conjunto *Covertype* cumple los requisitos de tamaño (581.012 celdas y 54 predictoras), calidad (sin faltantes, sin duplicados, sin valores imposibles ni códigos centinela) y licencia de uso (CC BY 4.0). Su principal característica es el desbalance: la clase mayoritaria (pino lodgepole) es 103 veces más frecuente que la minoritaria (álamo/sauce).

**Sobre el análisis exploratorio.**

1. **La elevación es la variable dominante.** Explica cerca de la mitad de la variabilidad entre clases (ε² = 0,51) y reproduce el gradiente altitudinal de la vegetación de montaña: el álamo/sauce y el pino ponderosa en las zonas bajas y el abeto/pícea y el *krummholz* cerca del límite superior del bosque.
2. **El área silvestre y el tipo de suelo están fuertemente asociados con la cobertura** (V de Cramér de 0,44 y 0,47) y entre sí (V = 0,69). Algunas clases están confinadas a un área: el álamo/sauce solo existe en Cache la Poudre.
3. **La iluminación forma un bloque redundante.** Los tres índices de sombreado dependen de la orientación y la pendiente, tienen VIF entre 38 y 144 y una asociación despreciable con la cobertura.
4. **Las dos clases mayoritarias se superponen** en todas las variables, y los datos no presentan grupos naturales que coincidan con la cobertura (ARI menor a 0,04). Las fronteras entre clases son no lineales.
5. **Los valores atípicos son legítimos** y se concentran en las clases de los extremos del gradiente altitudinal, por lo que se conservaron.

**Sobre la fuga de información y la estructura.** No hay identificadores, variables derivadas del objetivo ni variables posteriores a la predicción. La elevación y el área de Cache la Poudre superan un AUC univariado de 0,95 para algunas clases, pero por razones ecológicas y geográficas legítimas; quedan en observación. Los datos no tienen componente temporal ni coordenadas, y la dependencia espacial se evaluó a través del área silvestre.

**Sobre el modelo base.** La regresión logística multinomial, con ingeniería de variables y sin ponderación de clases, obtuvo en el conjunto de prueba:

| Métrica | Dummy (más frecuente) | Regresión logística | IC 95 % |
|---|---|---|---|
| Accuracy | 0,488 | 0,724 | 0,722–0,727 |
| Exactitud balanceada | 0,143 | 0,532 | 0,525–0,539 |
| F1 macro | 0,094 | 0,558 | 0,551–0,565 |
| AUC OvR macro | 0,500 | 0,937 | 0,936–0,938 |

El modelo supera ampliamente a la línea base trivial, está muy bien calibrado y no presenta sobreajuste: las métricas de entrenamiento, validación cruzada y prueba coinciden, y la curva de aprendizaje se estabiliza a partir de unas 30.000 celdas. Su limitación es de **sesgo**. No reconoce el álamo temblón (*recall* de 0,04), confunde el abeto/pícea con el pino lodgepole y comete más errores en los extremos del gradiente altitudinal. Un modelo no lineal de referencia alcanza 0,81 de *accuracy* y 0,74 de F1 macro con los mismos datos, lo que confirma que el problema no es trivial y que existe un margen amplio de mejora, que la Parte II cuantifica de forma sistemática.

## 4.2 Limitaciones

- **Representatividad geográfica.** Los datos cubren cuatro áreas silvestres del norte de Colorado con poca intervención humana. Cuando se deja fuera un área completa, el *accuracy* cae a 0,56 en promedio (0,53 para Cache la Poudre), de modo que el modelo no debe aplicarse a otras regiones sin datos locales.
- **Ausencia de coordenadas.** No fue posible medir la autocorrelación espacial (I de Moran, semivariogramas) ni construir bloques espaciales finos. La partición aleatoria puede ser ligeramente optimista incluso dentro de las áreas muestreadas.
- **Antigüedad de los datos.** La cartografía es anterior a 1998 y no refleja cambios posteriores por incendios, plagas o cambio climático.
- **Calidad de las etiquetas.** La cobertura proviene de un sistema de información del USFS, no de un inventario de campo celda por celda, y algunas unidades de suelo podrían incorporar información de la vegetación en su definición.
- **Compromiso entre clases.** Con la ponderación de clases el *recall* de las clases minoritarias mejora mucho (por ejemplo, el del álamo temblón pasa de 0,04 a 0,79), a costa del *accuracy* global. La Parte II resuelve este compromiso con el F1 macro como criterio de selección y con modelos no lineales capaces de reconocer las clases minoritarias sin sacrificar las mayoritarias.
