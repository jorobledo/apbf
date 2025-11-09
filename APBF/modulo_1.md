# 1. Introducción a la inteligencia artificial y al aprendizaje profundo

Hoy en día, la inteligencia artificial (IA) es un campo fértil con muchas aplicaciones prácticas y un área de investigación extremadamente activa. Se buscan soluciones de parte de softwares inteligentes para la automatización de rutinas de trabajo, para entender imágenes o comprender el habla, para hacer diagnósticos médicos, para asistir a la investigación científica básica, y mucho más.

Algunos éxitos tempranos de la IA se dieron en aplicaciones donde las computadoras o bien los modelos no requerían tener conocimiento acerca del mundo, sino más bien de una realidad acotada. Por ejemplo, en 1997 el modelo Deep Blue de IBM pudo vencer al campeón mundial de ajedrez del momento, Garry Kasparov. Este modelo se tuvo que entrenar en un espacio muy acotado de 64 casillas y 32 piezas que se pueden mover bajo determinadas reglas. Fue capaz de divisar estrategias de ajedrez basados en los movimientos del contrincante y fue un éxito rotundo del momento. Sin embargo, las reglas estaban bien definidas y la posibilidad de acciones en la realidad del ajedrez eran extremadamente acotadas. 

Al parecer, tareas formales y abstractas que son desafiantes y difíciles para los seres humanos suelen ser las más fáciles para las computadoras. Desde hace tiempo que una IA puede ganarle al mejor jugador de ajedrez del mundo, pero tareas super sencillas para nosotros, como reconocer objetos o entender frases y poder hablar, han sido desafíos enormes para las computadoras y recién hace relativamente poco han podido empezar a resolver estas tareas con capacidades similares que los seres humanos.

Han habido intentos de codificar a fuerza bruta el conocimiento acerca del mundo en lenguajes formales de programación, donde luego la computadora puede razonar automáticamente acerca de las declaraciones en este lenguaje formal infiriendo mediante reglas lógica. Este abordaje a la IA se conoce como abordaje basado en el conocimiento {cite}`goodfellow2016deep`, pero ninguno de estos proyectos ha resultado en éxitos mayores. Para los interesados, uno de los proyectos más famosos de esta índole fue `Cyc` {cite}`lenat1989building`, pero han habido claros contraejemplos de falta de entendimiento de las situaciones hipotéticas planteadas en el momento de inferencia. 

Sin embargo, las dificultades que aparecieron ante este intento de codificar a fuerza bruta el conocimiento sugirieron que los sistemas de inteligencia artificial debían tener la habilidad de adquirir su propio conocimiento a partir de la identificación de patrones en datos crudos. A esta capacidad se la llamó **Aprendizaje automático** y con su introducción las computadoras fueron capaces de abordar problemas que involucraban conocimiento del mundo real y tomar decisiones que aparentaban subjetivas. Así aparecieron modelos de aprendizaje automático como la regresión logística, o naive Bayes, máquinas de soporte vectorial, árboles de decisión, entre tantos otros.

La performance de estos algoritmos simples de aprendizaje automático depende fuertemente de la representación de los datos con los cuáles son entrenados. Cada pieza de información que se le da de entrada al modelo para su entrenamiento se conoce como una característica o *feature* en inglés. 

El aprendizaje automático ha experimentado una evolución notable en las últimas décadas, transformándose de un campo de investigación teórico a una herramienta esencial en diversas disciplinas, incluida la física. Este progreso ha sido impulsado por avances en el poder computacional, la disponibilidad de grandes volúmenes de datos y el desarrollo de algoritmos más sofisticados. Dentro del aprendizaje automático, se distinguen tres paradigmas principales: el aprendizaje supervisado, el no supervisado y el aprendizaje por refuerzo, cada uno con sus propias características y aplicaciones.

El **aprendizaje supervisado** es quizás el más intuitivo de los tres paradigmas. En este enfoque, los modelos son entrenados utilizando un conjunto de datos etiquetados, donde cada entrada está asociada con una salida deseada. Este método se asemeja a un proceso de enseñanza tradicional, donde un "maestro" proporciona ejemplos correctos y el "estudiante" (el modelo) aprende a generalizar a partir de ellos. Históricamente, el aprendizaje supervisado ha sido fundamental en tareas como la clasificación de imágenes, el reconocimiento de voz y la predicción de series temporales. Su evolución ha estado marcada por el desarrollo de algoritmos como las máquinas de soporte vectorial y, más recientemente, las redes neuronales profundas, que han permitido avances significativos en precisión y capacidad de generalización.

Por otro lado, el **aprendizaje no supervisado** aborda el desafío de encontrar patrones ocultos en datos no etiquetados. Este paradigma es especialmente relevante en situaciones donde la anotación de datos es costosa o imposible. A lo largo de los años, el aprendizaje no supervisado ha evolucionado desde técnicas básicas de agrupamiento, como el algoritmo k-means, hasta métodos más complejos como los autoencoders y las redes generativas adversarias (GANs). Estos avances han permitido aplicaciones innovadoras, como la reducción de dimensionalidad y la generación de datos sintéticos, ampliando las fronteras de lo que es posible en el análisis de datos.

Finalmente, el **aprendizaje por refuerzo** se inspira en la forma en que los seres vivos aprenden a través de la interacción con su entorno. En este enfoque, un agente aprende a tomar decisiones secuenciales mediante un proceso de prueba y error, recibiendo recompensas o castigos en función de las acciones que realiza. Este paradigma ha ganado prominencia en los últimos años, especialmente con el desarrollo de algoritmos de aprendizaje profundo por refuerzo que han logrado hazañas impresionantes, como vencer a campeones humanos en juegos complejos como el Go y el ajedrez. La evolución del aprendizaje por refuerzo refleja un cambio hacia sistemas más autónomos y adaptativos, capaces de aprender comportamientos complejos en entornos dinámicos.

En conjunto, estos tres paradigmas del aprendizaje automático han transformado nuestra capacidad para analizar y modelar datos, abriendo nuevas oportunidades para la investigación y la innovación en física y otras disciplinas científicas. A medida que continuamos explorando sus aplicaciones, es esencial comprender las fortalezas y limitaciones de cada enfoque, así como su evolución histórica, para aprovechar al máximo su potencial en la resolución de problemas complejos.

El **aprendizaje profundo** ha emergido como una subdisciplina del aprendizaje automático que ha revolucionado la forma en que abordamos problemas complejos de análisis de datos y modelado. Este enfoque se basa en el uso de redes neuronales artificiales con múltiples capas, conocidas como redes neuronales profundas, que son capaces de aprender representaciones jerárquicas de los datos. A diferencia de los métodos tradicionales de aprendizaje automático, que a menudo requieren de un preprocesamiento intensivo y la extracción manual de características, el aprendizaje profundo permite que los modelos descubran automáticamente las características relevantes directamente a partir de los datos brutos. Este avance ha sido posible gracias a mejoras en algoritmos de optimización, la disponibilidad de grandes conjuntos de datos y el incremento del poder computacional, especialmente a través del uso de unidades de procesamiento gráfico (GPUs). En los últimos años, el aprendizaje profundo ha demostrado ser extraordinariamente eficaz en una amplia gama de aplicaciones, desde el reconocimiento de imágenes y el procesamiento del lenguaje natural hasta la simulación de fenómenos físicos complejos. Al integrar el aprendizaje profundo con principios físicos, se abre un nuevo horizonte de posibilidades, permitiendo a los científicos y ingenieros desarrollar modelos más precisos y eficientes que respeten las leyes fundamentales de la naturaleza.

## Ejemplo de aprendizaje supervisado

Supongamos que queremos predecir la temperatura diaria de una ciudad en función de diversas características climáticas. Disponemos de un conjunto de datos históricos que incluye la temperatura máxima del día anterior, la humedad promedio, la velocidad del viento y la cantidad de precipitación. Nuestro objetivo es construir un modelo de aprendizaje automático supervisado que pueda predecir la temperatura máxima del día siguiente.

```python
# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar un conjunto de datos sintético
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 4)  # Características: temperatura anterior, humedad, viento, precipitación
y = 30 + 10 * X[:, 0] - 5 * X[:, 1] + 2 * X[:, 2] + np.random.randn(n_samples)  # Temperatura máxima

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse:.2f}")

# Mostrar los coeficientes del modelo
print("Coeficientes del modelo:", model.coef_)
```

```{bibliography} ./references.bib
:style: unsrt
```
