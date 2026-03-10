---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Cuantificación de incerteza en redes neuronales

```{contents}
:local:
```
Es importante tener presente que todas las mediciones, modelos y discretizaciones conllevan incertidumbre. En el caso de las mediciones y observaciones, esta incertidumbre suele manifestarse en forma de errores de medición. Por otro lado, las ecuaciones de los modelos generalmente describen solo una parte del sistema de interés, dejando el resto de los fenómenos no modelados como una fuente adicional de incertidumbre. Finalmente, en las simulaciones numéricas introducimos inevitablemente errores de discretización, asociados a la aproximación de ecuaciones continuas mediante métodos computacionales.

Frente a esta situación surge una pregunta fundamental: ¿cómo podemos saber si la respuesta que obtenemos es realmente la correcta?

Desde el punto de vista estadístico, una manera natural de abordar este problema consiste en considerar la distribución de probabilidad posterior. Esta distribución describe los distintos valores que podrían tomar los parámetros del modelo y captura las incertidumbres asociadas tanto a los datos como al modelo utilizado.

Este problema se vuelve aún más complejo en el contexto del aprendizaje automático, donde con frecuencia enfrentamos la tarea de aproximar funciones complejas y desconocidas. Desde una perspectiva probabilística, el procedimiento estándar de entrenamiento de una red neuronal produce típicamente una estimación por máxima verosimilitud (MLE) de los parámetros de la red.

Sin embargo, esta perspectiva basada en MLE no tiene en cuenta muchas de las incertidumbres mencionadas anteriormente. En el entrenamiento de modelos de aprendizaje profundo también intervienen procesos de optimización numérica, lo que implica la presencia de errores de aproximación e incertidumbre acerca de la representación que finalmente aprende el modelo.

Idealmente, el proceso de aprendizaje debería reformularse de manera tal que tenga en cuenta sus propias fuentes de incertidumbre y permita realizar inferencia posterior, es decir, que el modelo aprenda a producir una distribución completa sobre sus salidas en lugar de un único valor puntual. Sin embargo, lograr esto resulta ser una tarea extremadamente difícil.

Es en este contexto donde aparecen los llamados modelos de redes neuronales Bayesianas (Bayesian Neural Networks, BNN). Estos enfoques permiten realizar una forma de inferencia posterior al introducir distribuciones de probabilidad sobre los parámetros de la red neuronal. En lugar de considerar los pesos de la red como valores fijos, se los modela como variables aleatorias.

De esta manera obtenemos una distribución sobre los parámetros del modelo. Evaluando la red varias veces con distintos valores muestreados de esta distribución, podemos obtener diferentes realizaciones de la salida de la red, lo que permite aproximar la distribución de probabilidad de las predicciones.

Aun así, esta tarea sigue siendo muy desafiante. En general, entrenar una red neuronal Bayesiana es considerablemente más difícil que entrenar una red neuronal estándar. Esto no resulta sorprendente, ya que en este caso el objetivo es aprender una distribución de probabilidad completa, mientras que en los métodos tradicionales de aprendizaje profundo se aprende simplemente una estimación puntual de los parámetros. Incluso estas estimaciones puntuales ya representan problemas de optimización altamente complejos.


### Incertidumbre aleatoria vs. epistémica

Aunque no profundizaremos en detalle en este tema dentro del alcance de este libro, en muchos trabajos se distinguen dos tipos principales de incertidumbre que conviene mencionar:

- Incertidumbre aleatoria (aleatoric uncertainty), que corresponde a la incertidumbre inherente a los datos, por ejemplo el ruido presente en las mediciones.

- Incertidumbre epistémica (epistemic uncertainty), que describe la incertidumbre asociada al modelo, por ejemplo la incertidumbre sobre los parámetros de una red neuronal entrenada.

En las secciones siguientes nos centraremos principalmente en la incertidumbre epistémica, a través de métodos de inferencia posterior. Sin embargo, es importante tener en cuenta que en problemas reales distintos tipos de incertidumbre pueden aparecer simultáneamente, lo que hace que el análisis y la separación resulte mucho más difícil.

## Redes Neuronales Bayesianas (BNN)

## Monte Carlo Drop-out

## Ensambles profundos


```{bibliography}
:style: unsrt
:filter: docname in docnames
```