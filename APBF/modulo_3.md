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
# Embebiendo física dentro de redes neuronales


```{contents}
:local:
```

En los capítulos anteriores, hemos visto una introducción a las redes neuronales profundas, en donde observamos que encuentran soluciones a partir de los datos a los problemas presentados. Hemos observado además, que no pueden aprender estríctamente más allá de lo que los datos proveen de información y que cualquier extrapolación es, al igual que con cualquier otro tipo de modelo físico o estadístico, incapaz de proveer de un resultado razonable sin ningún tipo de auxilio o consideración especial para que esto suceda. No es magia, el modelo de aprendizaje automático está aprendiendo patrones *a partir* de los datos. Por ende, si somos capaces de conseguir más datos, tendremos mejor representación de la varianza del conjunto de datos y podremos conseguir un modelo que funcione mejor en mayor cantidad de circunstancias. Es por esto que los avances en aprendizaje profundo han sido fundamentales, ya que por más que se incrementasen las bases de datos, los modelos no eran capaces de ser entrenados en tiempos razonables con las computadoras existentes. 

Ahora, si queremos resolver un problema en particular en donde tenemos información extra acerca del sistema... por qué no usarla? ¿Por qué no restringir el espacio de posibles modelos a aquellos que cumplan directamente con reglas o leyes que conocemos? Estas preguntas ya se plantearon en otras áreas como en la estadística, en donde los modelos Bayesianos incluyen información *a priori* para realizar estimaciones de probabilidad *a posteriori* luego de ver los datos. Es decir, estamos sesgando las posibilidades de los modelos intencionalmente a aquellos a los que realmente nos interesan y tienen validez en el contexto del problema que se nos presenta. Volveremos a la estadística bayesiana en los capítulos 4 y 5. Por ahora, vamos a ver como hacer para que nuestros modelos de aprendizaje automático tengan en cuenta las leyes de la física con como las entendemos. 


## Redes neuronales basadas en la física (PINNs)

Las ideas detrás de las redes neuronales basadas en la física (PINNs, del inglés *Physics Informed Neural Networks*) se comenzaron a plantear en el 2017 con los trabajos de Raiisi y Karniadakis {cite}`raissi2017physicsinformeddeeplearning,raissi2017physicsinformeddeeplearning2`. Estas redes consistuyen una manera de integrar los conocimientos físicos explícitos expresados mediante ecuaciones diferenciales que gobiernan el sistema bajo estudio. Las PINNs incorporan leyes de conservación, ecuaciones constitutivas y condiciones iniciales y de frontera directamente en el proceso de entrenamiento. 

El principio fundamental de las PINNs consiste en representar la solución de un problema físico mediante una red neuronal y definir una función de pérdida que penaliza no sólo el error respecto a los datos observadors, sino también la falta de cumplimiento de las ecucaciones que gobiernan la física del problema. Las ecuaciones diferenciales vienen escritas en forma derivadas parciales de la solución $u$, que puede depender de varias variables, como la posición, velocidad, tiempo, etc, que apreviaremos $u(\vec{x})$. En el caso más sencillo de PINN, la red neuronal $q(\vec{x})$ aproxima la solución real del problema $u(\vec{x})$ a partir de datos observados. Gracias a la diferenciación automática vista en el capítulo anterior, resulta posible calcular las derivadas parciales de la red neuronal $q$ con respecto a sus variables de entrada $\vec{x}$, es decir, resulta posible calcular el gradiente $\nabla_{\vec{x}}{q(\vec{x})}$. Este gran hecho permite evaluar residuos de ecuaciones diferenciales ordinarias o parciales sin necesidad de discretizaciones como mallas o métodos de elementos finitos. Este enfoque resulta particularmente atractivo en escenarios en donde los datos son escasos, ruidosos o incompletos, pero se dispone de modelos físicos bien establecidos.

Las PINNs han demostrado utilidad en una amplia variedad de aplicaciones, incluyendo dinámica de fluídos, transferencia de calor, mecánica de sólidos, electromagnetismo, e incluse en sistemas biológicos. Éstas ofrecen una formulación unificada para problemas directos e inversos, permitiendo tanto la estimación de campos físicos como la identificación de parámetros desconocidos. 


## Diferenciación automática para ecuaciones diferenciales parciales con constraints
