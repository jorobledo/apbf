# Apéndice: Introducción a Python

## Librerías principales que se utilizan en estas notas

### Numpy

NumPy (Numerical Python) es una biblioteca fundamental para el cálculo numérico en Python, especialmente relevante para estudiantes de física y disciplinas afines. Proporciona soporte para arrays y matrices multidimensionales, junto con una colección de funciones matemáticas de alto nivel para operar con estos datos. NumPy es esencial para realizar cálculos eficientes y es la base sobre la cual se construyen muchas otras bibliotecas científicas en Python, como SciPy, Pandas y Matplotlib.

#### Arrays y Operaciones Básicas

En el corazón de NumPy se encuentra el objeto ndarray, que representa un array n-dimensional. A diferencia de las listas de Python, los arrays de NumPy son más eficientes en términos de memoria y permiten realizar operaciones matemáticas de manera vectorizada, lo que significa que las operaciones se aplican a todos los elementos del array simultáneamente, sin necesidad de bucles explícitos.

```python
import numpy as np

# Crear un array unidimensional
a = np.array([1, 2, 3, 4, 5])

# Crear un array bidimensional (matriz)
b = np.array([[1, 2, 3], [4, 5, 6]])

# Operaciones básicas
suma = a + 10  # Sumar 10 a cada elemento
producto = a * 2  # Multiplicar cada elemento por 2
```

#### Funciones Matemáticas
NumPy ofrece una amplia gama de funciones matemáticas que se aplican de manera eficiente a los arrays. Esto incluye funciones trigonométricas, exponenciales, logarítmicas y estadísticas.

```python
# Funciones matemáticas
sin_a = np.sin(a)  # Seno de cada elemento
log_b = np.log(b)  # Logaritmo natural de cada elemento

# Estadísticas básicas
media = np.mean(a)  # Media de los elementos
desviacion = np.std(a)  # Desviación estándar
```

#### Manipulación de Arrays
NumPy permite manipular arrays de diversas maneras, como cambiar su forma, apilarlos, dividirlos y transponerlos.

```python 
# Cambiar la forma de un array
c = np.arange(12)  # Crear un array de 0 a 11
c_reshaped = c.reshape(3, 4)  # Cambiar la forma a 3x4

# Apilar arrays
d = np.array([6, 7, 8])
apilado = np.vstack((a, d))  # Apilar verticalmente

# Transponer un array
b_transpuesta = b.T
```

#### Indexación y Slicing
La indexación y el slicing en NumPy son similares a las listas de Python, pero con mayor flexibilidad para arrays multidimensionales.

```python 
# Indexación
elemento = b[1, 2]  # Elemento en la segunda fila, tercera columna

# Slicing
sub_array = b[:, 1:3]  # Todas las filas, columnas 1 a 2
```


### Pandas

Pandas es una biblioteca de Python diseñada para facilitar el análisis y la manipulación de datos, y es especialmente útil para estudiantes de física que trabajan con grandes conjuntos de datos experimentales o simulados. Construida sobre NumPy, Pandas proporciona estructuras de datos y funciones de alto nivel que simplifican tareas comunes de manipulación de datos, como la limpieza, transformación y agregación.

#### Estructuras de Datos Principales
Pandas introduce dos estructuras de datos fundamentales: Series y DataFrame.

-  Series: Es una estructura unidimensional similar a un array, lista o columna en una tabla. Cada elemento en una Series tiene un índice asociado, lo que permite un acceso más flexible a los datos.

```python 
import pandas as pd

# Crear una Series
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
```

- DataFrame: Es una estructura bidimensional, similar a una hoja de cálculo o una tabla SQL, que contiene filas y columnas. Los DataFrames son el núcleo de Pandas y permiten almacenar y manipular datos tabulares de manera eficiente.

```python
# Crear un DataFrame
data = {'Temperatura': [22, 21, 19, 23],
        'Humedad': [30, 45, 50, 40]}
df = pd.DataFrame(data, index=['Día 1', 'Día 2', 'Día 3', 'Día 4'])
print(df)
```

#### Manipulación de Datos

Pandas ofrece una amplia gama de funciones para manipular y transformar datos, lo que facilita tareas comunes en el análisis de datos.

- **Selección y Filtrado**: Puedes seleccionar columnas, filas o valores específicos utilizando etiquetas o condiciones.

```python
# Seleccionar una columna
temperatura = df['Temperatura']

# Filtrar filas basadas en una condición
alta_humedad = df[df['Humedad'] > 40]
```

- **Operaciones de Agregación**: Pandas permite realizar operaciones de agregación como suma, media y conteo de manera sencilla.

```python
# Calcular la media de cada columna
media_columnas = df.mean()

# Sumar todos los valores de una columna
suma_temperatura = df['Temperatura'].sum()
```

- **Manejo de Datos Faltantes**: Pandas proporciona métodos para identificar y manejar datos faltantes, lo cual es crucial en el análisis de datos reales.

```python
# Identificar valores faltantes
faltantes = df.isnull()

# Rellenar valores faltantes
df_rellenado = df.fillna(0)
```

#### Operaciones Avanzadas

- **Agrupación**: La función `groupby` permite agrupar datos y aplicar funciones de agregación a cada grupo.

```python
# Agrupar por una columna y calcular la media
grupo = df.groupby('Humedad').mean()
```

- **Combinación de Datos**: Pandas facilita la combinación de múltiples `DataFrames` mediante operaciones de concatenación y fusión.

```python
# Concatenar DataFrames
df_concatenado = pd.concat([df, df])

# Fusionar DataFrames
df_otra = pd.DataFrame({'Humedad': [30, 45], 'Presión': [1012, 1015]})
df_fusionado = pd.merge(df, df_otra, on='Humedad')
```

### Scikit-learn

### Pytorch

### Matplotlib
