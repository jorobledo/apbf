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
# Introducción al aprendizaje profundo basado en la física

```{contents}
:local:
```

## De modelos basados en los datos a basados en la física

El aprendizaje profundo ha demostrado ser una herramienta poderosa para abordar problemas complejos en diversas áreas, desde la visión por computadora hasta el procesamiento del lenguaje natural. Sin embargo, a medida que se amplían las aplicaciones de estas técnicas, surge la necesidad de integrar conocimientos específicos de dominio en los modelos para mejorar su precisión y eficiencia. En el contexto de la física y otras ciencias naturales, los modelos basados únicamente en datos pueden no capturar adecuadamente las leyes fundamentales que rigen los fenómenos estudiados. Aquí es donde entran en juego las Redes Neuronales Basadas en la Física (PINNs, por sus siglas en inglés).

Las PINNs representan un enfoque innovador que combina el poder del aprendizaje profundo con principios físicos bien establecidos. A diferencia de los modelos tradicionales de aprendizaje profundo, que dependen exclusivamente de grandes volúmenes de datos para aprender patrones, las PINNs incorporan ecuaciones diferenciales parciales (PDEs) y otras leyes físicas directamente en el proceso de entrenamiento. Esto no solo mejora la capacidad del modelo para generalizar a partir de datos limitados, sino que también garantiza que las predicciones respeten las restricciones físicas inherentes al problema. La motivación para avanzar hacia las PINNs radica en su potencial para resolver problemas complejos en áreas como la dinámica de fluidos, la mecánica cuántica y la biología computacional, donde las simulaciones precisas y eficientes son cruciales. Al integrar el conocimiento físico en el aprendizaje profundo, las PINNs ofrecen una nueva perspectiva para abordar desafíos científicos, permitiendo modelos que no solo son precisos, sino también interpretables y consistentes con las leyes de la naturaleza.

En el aprendizaje profundo basado en la física, se extiende la idea de función de costo de manera natural. En este caso, se combinan términos asociados a los datos con términos que miden el incumplimiento de las leyes físicas, típicamente expresadas mediante ecuaciones diferenciales, condiciones iniciales y de frontera, o principios variacionales. De forma esquemática, la función de costo adopta la forma 

$$
\mathcal{L} = \mathcal{L}_{\text{datos}} + \lambda \mathcal{L}_{\text{física}}
$$

donde el segundo término penaliza las desviaciones respecto a las ecucaciones que gobiernan el sistema y $\lambda$ controla elbalance entre ajuste a los datos y la consistencia física de la solución. 

Este enfoque unifica el aprendizaje automática con el modelado físico en un único marco de optimización, en el que las redes neuronales actúan como aproximadores universales, mientras que la función de costo garantiza que las soluciones aprendidas respeten la estructura fundamental del problema en cuestión. 


## Evolución de aprendizaje automático a aprendizaje profundo


El aprendizaje profundo es una subdisciplina del aprendizaje automático que ha revolucionado la forma en que abordamos problemas complejos de análisis de datos y modelado. Se caracteriza por el uso de redes neuronales artificiales con múltiples capas, conocidas como redes neuronales profundas, que son capaces de aprender representaciones jerárquicas de los datos. A diferencia de los métodos tradicionales de aprendizaje automático, que a menudo requieren de un preprocesamiento intensivo y la extracción manual de características, el aprendizaje profundo permite que los modelos descubran automáticamente las características relevantes directamente a partir de los datos brutos. Este avance ha sido posible gracias a mejoras en algoritmos de optimización, la disponibilidad de grandes conjuntos de datos y el incremento del poder computacional, especialmente a través del uso de unidades de procesamiento gráfico (GPUs).

El término "profundo" en aprendizaje profundo se refiere a la presencia de múltiples capas ocultas en una red neuronal, lo que permite al modelo aprender características de alto nivel de los datos de manera progresiva. Históricamente, el concepto de redes neuronales se remonta a mediados del siglo XX, pero fue en la última década que el aprendizaje profundo ganó prominencia, impulsado por avances como las redes neuronales convolucionales (CNNs) para el procesamiento de imágenes y las redes neuronales recurrentes (RNNs) para el procesamiento del lenguaje natural. Modelos como AlexNet, que ganó el concurso ImageNet en 2012, demostraron el poder de las CNNs para tareas de visión por computadora, mientras que las RNNs y sus variantes, como las redes LSTM, han sido fundamentales para el análisis de secuencias temporales. Más recientemente, los transformadores han revolucionado el campo del procesamiento del lenguaje natural, permitiendo el desarrollo de modelos como BERT y GPT, que han establecido nuevos estándares en tareas de comprensión y generación de texto. Estos modelos han ampliado las fronteras de lo que es posible en inteligencia artificial, permitiendo aplicaciones innovadoras en áreas como la visión por computadora, el reconocimiento de voz y la traducción automática.

En lo que continúa revisaremos algunos de los conceptos para mí más importantes que permitieron avanzar del aprendizaje automático al aprendizaje profundo y que nos serán de utilidad para entender los modelos que plantearemos más adelante.

### Diferenciación Automática y retropropagación

Como he mencionado anteriormente, el entrenamiento de modelos de aprendizaje automático profundo se basa en el cálculo eficiente del gradiente de la función (escalar) de costo $\mathcal{L}(\theta)$ con respecto a un gran número de parámetros $\theta$, i.e. $\theta \in \mathbf{R}^n$ con $n$ grande. Los algoritmos de optimización requieren el gradiente $\nabla_{\theta}\mathcal{L}$ para actualizar los parámetros de forma iterativa, pero el cálculo manual de estos gradientes resulta impracticable para modelos reales. 

La **diferenciación automática** (Autograd) es una técnica computacional que permite calcular derivadas exactas (hasta la precisión de la máquina) de funciones definidas programáticamente de manera eficiente y sistemática. A diferencia de la diferenciación simbólica o de la numérica, la automática o autograd se basa en la aplicación repetida de la regla de la cadena sobre operaciones elementales.

La base del Autograd es el grafo computacional. Cualquier función implementada como una secuencia de operaciones elementales (suma, producto, exponencial, etc.) puede representarse como un grafo dirigido acíclico donde los nodos representan operaciones o variables intermedias y las aristas representan dependencias funcionales. Por ejemplo, $f(x,y)=xy + sin(x)$ puede considerarse como $f(a,b)=z=a+b$ con $a=xy$ y $b=sin(x)$ y el grafo puede ser representado como se muestra en la siguiente figura.

![](./figs/graph.png)

Si aplicamos la regla de la cadena para una variable $t$ (que puede ser $x$ o $y$), podemos escribir lo siguiente

$$
\frac{\partial z}{\partial t} = \frac{\partial a}{\partial t} + \frac{\partial b}{\partial t}
$$

con 

$$
\frac{\partial b}{\partial t} = \cos(x) \frac{\partial x}{\partial t} 
$$

y

$$
\frac{\partial a}{\partial t} = x\frac{\partial y}{\partial t} + y\frac{\partial x}{\partial t}
$$

Si escribimos estas expresiones en un programita que involucra variables diferenciales $(dx, dy)$ nos queda

```python
da = y * dx + x*dy
db = cos(x) * dx
dz = da + db
```

Si queremos saber la derivada para $x$ basta con remplazar $dx=1$ y $dy=0$ y viceversa para la derivada con respecto a $y$.  Así empiezan a aparecer unas reglas básicas programáticas, como por ejemplo, si $c=a+b$ (suma) entonces $dc=da+db$. Si $c=a*b$, entonces $dc=b*da + a*db$, y si $c=sin(a)$ entonces $dc=cos(a)*da$. De la misma manera, podemos armarnos de las reglas para la resta, la división, la exponenciación, el coseno y la tangente (entre otras más).

La diferenciación automática se puede hacer de dos maneras, hacia adelante o hacia atrás (retropropagación) pero a continuación daré argumentos de por qué se utiliza la diferenciación hacia atrás en aprendizaje profundo. Hacia adelante sería: El programa lee $x$ y $dx$, $y$ y $dy$. En este ejemplo $a=x*y$, por lo que con las reglas que obtuvimos $da=y*dx + x*dy$. Como $b=sin(x)$, por la regla que obtuvimos $db=cos(x) +dx$. Finalmente, como $z=a+b$, por la regla de la suma $dz=da+db$. De esta manera, si $dx=1$ y $dy=0$, se obtiene la derivada de la función $z$ con respecto a $x$ computacionalmente. A pesar de que la diferenciación automática hacia adelante resulta muy fácil de implementar y entender, presenta una gran desventaja computacional. Para CADA variable sobre la que queremos calcular el gradiente, debemos correr el programa que realiza toda la computación desde el inicio hasta el fin. Esto claramente representa un problema gigante para las redes neuronales profundas (que pueden presentar más de millones de parámetros).

La solución a este problema es la diferenciación automática hacia atrás o bien la retropropagación del gradiente y se basa en revertir la regla de la cadena. En este caso, en vez de preguntarnos cómo varía la salida en función de las variables de entrada, nos preguntamos qué variables de salida pueden ser afectadas por una variable entrada dada. Así como usamos $t$ para expresar una variable genérica de entrada anteriormente, usando $s$ para una variable genérica de salida se puede escribir 

$$
\frac{\partial s}{\partial b} = \frac{\partial z}{\partial b}\frac{\partial s}{\partial z} = \frac{\partial s}{\partial z}
$$

$$
\frac{\partial s}{\partial a} = \frac{\partial z}{\partial a}\frac{\partial s}{\partial z} = \frac{\partial s}{\partial z}
$$

$$
\frac{\partial s}{\partial y} = \frac{\partial a}{\partial y}\frac{\partial s}{\partial a} = x\frac{\partial s}{\partial a}
$$

$$
\frac{\partial s}{\partial x} = \frac{\partial a}{\partial x}\frac{\partial s}{\partial a} + \frac{\partial b}{\partial x}\frac{\partial s}{\partial b}=y\frac{\partial s}{\partial a} + cos(x)\frac{\partial s}{\partial b}=(y+cos(x))\frac{\partial s}{\partial z}
$$

Escribiendo lo anterior en un programita, esto nos queda

```python
gb = gz
ga = gz
gy = x * ga
gx = y * ga + cos(x) * gb
```

Sustituyendo $s=z$ sería equivalente a $gz=1$, podemos ver entonces que para calcular ambos gradientes $\frac{\partial z}{\partial x}$ y $\frac{\partial z}{\partial y}$ debemos ejecutar el programa **una sola vez**.


Continuar...

### Conexiones residuales

A medida que las redes neuronales profundas comenzaron a escalar en número de capas, se observó un fenómeno paradójico. Al incrementar la profundidad del modelo, el desempeño en entrenamiento y validación podía verse afectado negativamente. Aún cuando el modelo más profundo tenía mayor capacidad expresiva que uno más superficial, no se osbervaban mejoras. Y esto no se debía al sobreajuste sino a una dificultad inherente en el proceso de optimización. 

Una de las causas principales de este fenómeno fue la atenuación o explosión del gradiente durante la retropropagación. En redes neuronales muy profundas, los gradientes pueden volverse extremadamente pequeños o grandes al atravesar muchas capas consecutivas, lo que dificulta el ajuste efectivo de los parámetros de las primeras capas. Las primeras ideas para mitigar este problema fueron la normalización de los datos por batches y comenzar con una inicialización adecuada de los pesos del modelo. A pesar de que esto tuvo un efecto positivo sobre el aprendizaje de los modelos profundo, no resultó suficiente para permitir entrenamiento estable de redes extremadamente profundas. 

lo que permitió destrabar este problema fueron las conexiones residuales, que fueron introducidas originalmente en las redes ResNet {cite}`he2015deepresiduallearningimage`. La idea principal consiste en reformular la red en bloques. En lugar de aprender directamente una transformación deseada $H(x)$ de la entrada $x$, cada bloque aprende una función residual $F(x)$ definida como 

$$
F(x) = H(x) -x,
$$

de modo que la salida del bloque se expresa como

$$
y = x + F(x)
$$

En términos prácticos, la implementación es una conexión directa que salta una o más capas (fijarse que y=x es parte de la transformación) y suma la entrada del bloque con su salida transformada. Efectivamente es incluir la identidad a la transformación lo cual no introduce parámetros adicionales, pero permite que no desaparezca el gradiente. 

Desde el punto de vista de optimización, las conexiones residuales facilitan el aprendizaje al permitir que el modelo ajuste perturbaciones pequeñas alrededor de la identidad. Si la transformación óptima es cercana a la identidad, el modelo la aproxima más facilmente aprendiendo un residual cercano a cero, en vez de forzar a las capas a aprender explícitamente una identidad completa mediante funciones no lineales. 

Además, las conexiones residuales crean cortecamino para el flujo del gradiente durante la retropropagación. Al derivar la salida de un bloque con respecto a su entrada, se obtiene

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x},
$$

donde $I$ es la identidad. Este término garantiza que, incluso si el gradiente asociado a $F(x)$ se atenúa, existe siempre una contribución directa que preserva la señal del gradiente a lo largo de la red. Como consecuencia, el gradiente se propaga de manera más estable hacia las capas más profundas (es decir, a las primeras capas ya que estamos retropropagando el gradiente).


La introducción de conexiones residuales marcó un punto de inflexión en el desarrollo del aprendizaje profundo. Esto condujo a mejoras directas y sustanciales en tareas de visión por computadora, reconocimiento de patrones y modelado de sistemas complejos.  Más allá de su formulación original, el rpincipio residual ha sido adoptado y extendido en múltiples arquitecturas modernas, incluyendo redes densas, transformadores y modelos utilizandos en aprendizajo profundo basado en la física. 


### Clasificación de imágenes 

La clasificación de imágenes constitute una de las aplicaciones más representativas y exitosas del aprendizaje profundo. El problema consiste en asignar a una imagen una etiqueta discreta que describe su contenido, como la presencia de un objeto, un estado físico o una categoría predefinida. Desde un punto de vista formal, una imagen puede interpretarse como una función discreta definida sobre una grilla bidimensional, cuyos valores representan intensidades, colores o magnitudes físicas medidas en el espacio.

Antes de la irrupción del aprendizaje profundo, los métodos de clasificación de imágenes dependían en gran medida de la extracción manual de características, tales como bordes, texturas o descriptores geométricos diseñados específicamente para cada dominio. Estos enfoques requerían un conocimiento experto considerable y presentaban limitaciones importantes al enfrentarse a variaciones complejas en escala, orientación, ruido o condiciones de adquisición.

El aprendizaje profundo transformó radicalmente este panorama mediante el uso de redes neuronales profundas capaces de aprender representaciones jerárquicas directamente a partir de los datos. En particular, las redes neuronales convolucionales introdujeron una estructura inductiva que explota la localización espacial y la invariancia traslacional inherentes a las imágenes. A través de la composición de múltiples capas convolucionales y no lineales, estos modelos aprenden progresivamente características de bajo nivel, como bordes y contrastes, y las combinan para formar descriptores de alto nivel relevantes para la tarea de clasificación.

Desde el punto de vista matemático, la clasificación de imágenes se formula como un problema de optimización en el que se busca ajustar los parámetros del modelo para minimizar una función de costo que mide la discrepancia entre las etiquetas verdaderas y las predicciones del modelo. El entrenamiento se realiza mediante retropropagación y optimización basada en gradientes, apoyándose en técnicas como normalización, regularización y arquitecturas profundas para garantizar estabilidad y buen desempeño.

En el contexto del aprendizaje profundo basado en la física, la clasificación de imágenes adquiere un significado adicional. En muchas aplicaciones científicas, las imágenes no son meramente representaciones visuales, sino mediciones de campos físicos, distribuciones de energía, densidades de probabilidad o estados del sistema bajo estudio. Ejemplos típicos incluyen imágenes médicas, patrones experimentales, simulaciones numéricas de campos físicos y datos obtenidos mediante sensores especializados.

### Modelos de regresión 

Los problemas de regresión constituyen una de las tareas fundamentales del aprendizaje automático y del aprendizaje profundo. En este tipo de problemas, el objetivo es aproximar una relación funcional entre un conjunto de variables de entrada y una o más variables de salida continuas. A diferencia de la clasificación, donde se asignan etiquetas discretas, la regresión busca predecir magnitudes reales que suelen representar cantidades físicas, estados del sistema o variables observables de interés.

El aprendizaje profundo amplía de manera significativa el alcance de los modelos de regresión clásicos mediante el uso de redes neuronales profundas como aproximadores universales de funciones. Al componer múltiples capas no lineales, estas redes son capaces de capturar dependencias complejas entre variables de entrada y salida, sin necesidad de especificar explícitamente la forma funcional de la relación subyacente. Desde un punto de vista matemático, el problema se formula como la minimización de una función de costo que mide el error entre las predicciones del modelo y los valores continuos observados, típicamente mediante métricas como el error cuadrático medio u otras funciones robustas.

En aplicaciones científicas y de ingeniería, los modelos de regresión con aprendizaje profundo se utilizan para aproximar campos físicos, resolver problemas inversos, emular simulaciones numéricas costosas y predecir la evolución temporal de sistemas dinámicos. En muchos de estos casos, las salidas del modelo no son simples valores escalares, sino funciones continuas en el espacio y el tiempo, lo que resalta el carácter funcional de la regresión en este contexto.

### Redes Neuronales Recurrentes

Muchos problemas relevantes en ciencia e ingeniería no pueden describirse adecuadamente como relaciones estáticas entre variables, sino que involucran dependencias temporales y evolución dinámica. En estos escenarios, los datos se presentan naturalmente como secuencias, ya sea en el tiempo, en el espacio o en ambos. Las redes neuronales recurrentes (RNN, por sus siglas en inglés) surgen como una extensión del aprendizaje profundo diseñada específicamente para modelar este tipo de estructuras secuenciales.

A diferencia de las redes neuronales feedforward, que procesan cada entrada de manera independiente, las RNN incorporan un estado interno que actúa como una memoria dinámica. Este estado permite que la salida del modelo en un instante dado dependa no solo de la entrada actual, sino también del historial previo. Desde un punto de vista matemático, una RNN define una relación recursiva en la que el estado oculto se actualiza en cada paso temporal mediante una transformación no lineal que combina la entrada presente con el estado anterior.

El entrenamiento de redes neuronales recurrentes se realiza mediante una extensión de la retropropagación conocida como retropropagación a través del tiempo. Este procedimiento desenrolla la recurrencia a lo largo de la dimensión temporal y permite calcular gradientes con respecto a los parámetros compartidos en todos los pasos. Sin embargo, la naturaleza recursiva de estas arquitecturas introduce desafíos adicionales, como la atenuación y explosión de gradientes, que limitan la capacidad de las RNN clásicas para capturar dependencias de largo alcance.

Para abordar estas dificultades, se desarrollaron variantes como las redes de memoria a largo y corto plazo (LSTM) y las unidades recurrentes con compuertas (GRU). Estas arquitecturas introducen mecanismos de control que regulan el flujo de información y gradientes a lo largo del tiempo, permitiendo aprender dependencias temporales más largas y estables. Gracias a estas mejoras, las redes neuronales recurrentes se han convertido en herramientas fundamentales para el modelado de series temporales, señales dinámicas y procesos secuenciales complejos.


## Sesgos inductivos y constraints físicos

Todo modelo de aprendizaje automático incorpora, de manera explícita o implícita, un conjunto de supuestos sobre el tipo de funciones que es capaz de aprender. Estos supuestos, conocidos como sesgos inductivos, determinan cómo el modelo generaliza más allá de los datos observados y juegan un papel central cuando la información disponible es limitada, ruidosa o incompleta.

En aprendizaje profundo, los sesgos inductivos se introducen principalmente a través de la arquitectura del modelo, la función de costo y el procedimiento de optimización. Por ejemplo, las redes convolucionales imponen un sesgo de localidad y de invariancia traslacional; las redes recurrentes introducen un sesgo temporal; y las conexiones residuales favorecen transformaciones cercanas a la identidad. Estos sesgos no garantizan por sí mismos que el modelo aprenda la “función correcta”, pero restringen el espacio de hipótesis de una manera que hace el aprendizaje más eficiente y estable.

En muchos problemas científicos y de ingeniería, además de los datos, se dispone de un conocimiento sustancial sobre el sistema bajo estudio. Este conocimiento se expresa típicamente en forma de leyes físicas, como ecuaciones diferenciales, principios de conservación, simetrías, restricciones de positividad o límites energéticos. Ignorar esta información y entrenar modelos puramente basados en datos no solo resulta ineficiente, sino que puede conducir a soluciones que, aunque ajusten los datos observados, violan principios fundamentales y carecen de significado físico.
```{contents}
:local:
```
