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
# Modelos probabilísticos

```{contents}
:local:
```

## Aprender distribuciones 

En muchas áreas de la física, el objetivo central no es únicamente predecir un valor específico, sino comprender y modelar la incertidumbre inherente a los sistemas naturales. Los fenómenos físicos suelen estar influenciados por múltiples fuentes de variabilidad: ruido experimental, condiciones iniciales desconocidas, interacciones microscópicas no observables o incluso naturaleza intrínsecamente estocástica del sistema.

Un modelo probabilístico permite representar el estado de un sistema físico mediante una distribución de probabilidad en lugar de un único valor determinista. Esto resulta especialmente natural en física estadística, mecánica cuántica y termodinámica, donde las magnitudes de interés se describen en términos de distribuciones y promedios. Por ejemplo, en lugar de predecir exactamente la posición de una partícula, se modela la probabilidad de encontrarla en una determinada región del espacio.

La motivación principal de los modelos probabilísticos radica en su capacidad para incorporar conocimiento previo del sistema, formular hipótesis físicas y actualizar dichas creencias a partir de datos experimentales. A partir de observaciones, el aprendizaje de una distribución de probabilidad permite capturar patrones, correlaciones y estructuras subyacentes que no son evidentes a nivel determinista. De esta manera, los modelos probabilísticos proporcionan un marco unificado para inferencia, predicción y toma de decisiones, manteniendo una estrecha conexión con los principios fundamentales de la física.

La motivación para utilizar modelos probabilísticos va más allá de la física. En disciplinas como la biología, la economía, la robótica o el aprendizaje automático, los datos suelen ser incompletos, ruidosos y de alta dimensionalidad. En estos escenarios, los enfoques deterministas resultan insuficientes para capturar la complejidad de los procesos subyacentes. Los modelos probabilísticos ofrecen un marco flexible para representar múltiples posibles resultados, cuantificar la incertidumbre de las predicciones y generar muestras coherentes con los datos observados.

Desde una perspectiva moderna, muchos de estos modelos se implementan mediante redes neuronales profundas, lo que permite escalar el aprendizaje de distribuciones de probabilidad a espacios de alta dimensión. A continuación, se presentan distintas familias de modelos probabilísticos ampliamente utilizados en aprendizaje automático generativo, cada una con principios y compromisos diferentes entre expresividad, eficiencia computacional e interpretabilidad.


## Diseñar una función de costo adecuada

El límite inferior de la evidencia (ELBO, del inglés *evidence lower bound*) es una cantidad importante que permite el entrenamiento de modelos probabilísticos y que veremos en detalle a continuación. 

En un modelo de variable latente, suponemos que nuestros dato observado $x$ es una realización de una variable aleatoria $X$. Más aún, pensamos que existe otra variable aleatoria $Z$ cuya probabilidad de distribución conjunta está determinada por $p(X,Z;\theta)$, en donde $\theta$ son los parámetros que parametrizan la distribución. Desafortunadamente, nuestros datos medidos u observados son sólo una realización de $X$ y no de $Z$, por lo que $Z$ permanece no observada o latente. 

Supongamos que queremos calcular la distribución de probabilidad posterior $p(Z|X;\theta)$ dado algún valor fijo de $\theta$. Es decir, cuál es la probabilidad de observar $Z=z$ dado que se observó $X=x$ para todo valor de $z$ y $x$. Este problema se puede plantear utilizando lo que se conoce como *inferencia variacional* y que describiremos en breves. 

Ahora supongamos que no conocemos $\theta$, pero queremos encontrar el estimador de máxima verosimilitud de $\theta$, i.e. $\text{argmax}_{\theta} l(\theta)$, en donde $l(\theta)$ es la función de log-verosimilitud, definida como

$$
l(\theta) := \log (p(x;\theta)) = \log \int_z p(x,z;\theta)dz.
$$
Este problema se puede plantear utilizando la *maximicación del valor esperado* o esperanza. 

Tanto la inferencia variacional como la maximización del valor esperado se basan en el límite inferior de la evidencia ELBO. 

Para entender ELBO hay que primero definir qué es la evidencia. La *evidencia* es el nombre que se le da a la función de verosimilitud evaluada en un vector de parámetros $\theta$ fijo. Esto se denota poniendo a $\theta$ del lado derecho del símbolo $;$ en la expresión

$$
\text{evidencia}:= \log p(x;\theta).
$$

Intuitivamente, si elegimos el modelo correcto para $p$ y $\theta$, esperaríamos que la probabilidad marginal de observar los datos $x$ que observamos sea muy alta. Por lo tanto, un valor alto para $\log p(x;\theta)$ indica de cierta manera que estamos en la dirección correcta al haber seleccionado el modelo $p$ y los parámetros $\theta$ para estos datos. Es decir, que esta cantidad es una "evidencia" que hemos elegido el modelo correcto para los datos.

Ahora supongamos que $Z$ sigue una distribución denotada por $q$. Utilizando el teorema de Bayes {cite}`wiki:Bayes` podemos escribir a la probabilidad conjunta $p(x,z;\theta)=p(x|z;\theta)q(z)$. El límite inferior para la evidencia no es más que

$$
\begin{align}
\log p(x;\theta)
&= \log \int p(x,z;\theta)\,dz \\
&= \log \int q(z)\,\frac{p(x,z;\theta)}{q(z)}\,dz \\
&= \log \mathbb{E}_{Z\sim q}
\left[
\frac{p(x,Z;\theta)}{q(Z)}
\right] \\
&\ge \mathbb{E}_{Z\sim q}
\left[
\frac{p(x,Z;\theta)}{q(Z)}
\right] =\mathbb{E}_{Z\sim q}
\left[
\log p(x,Z;\theta) - \log q(Z)
\right],
\end{align}
$$
en donde para pasar a la desigualdad hemos utilizado la desigualdad de Jensen {cite}`wiki:jensen`. Definimos entonces 

$$
ELBO := \left[
\frac{p(x,Z;\theta)}{q(Z)}
\right].
$$

Resulta que la diferencia entre la evidencia y el ELBO es exactamente la divergencia de Kullback-Leibler (KL) entre $p(z|x;\theta)$ y $q(z)$! Esto se puede probar partiendo desde la definición de la divergencia de Kullback-Leibler:

$$
\begin{align}
KL(q(z)||p(z|x;\theta)) :&= \mathbb E_{Z \sim q}\left[\log \frac{q(Z)}{p(Z|x;\theta)}\right] \\
& = \mathbb E_{Z \sim q}\left[\log q(Z)\right] - \mathbb E_{Z \sim q}\left[\log p(Z|x;\theta)\right] \\
& = \mathbb E_{Z \sim q}\left[\log q(Z)\right] - \mathbb E_{Z \sim q}\left[\log \frac{p(x,Z;\theta)}{p(x;\theta)}\right] \\
& = \mathbb E_{Z \sim q}\left[\log q(Z)\right] - \mathbb E_{Z \sim q}\left[\log p(x,Z;\theta)\right] + \mathbb E_{Z \sim q}\left[\log p(x;\theta)\right] \\
& = \log p(x;\theta) - \mathbb E_{Z \sim q}\left[\frac{\log p(x,Z;\theta)}{\log q(Z)}\right] \\
& = \text{evidencia} - ELBO.
\end{align}
$$

Esta resulta una identidad clave, ya que ahora podemos decir con exactitud que la evidencia es la suma del ELBO + la KL. Cuando el objetivo es maximizar la evidencia, resulta factible entonces maximizar el ELBO y de esta manera la KL va a medir cuán lejos está la aproximación $q(z)$ de la verdadera probabilidad posterior. Además vemos que Maximizar el ELBO equivale a minimizar la KL.

### Inferencia Variacional

La inferencia variacional sirve para estimar una distribución posterior cuando calcularla explícitamente resulta inviable. Como habíamos mencionado, queremos calcular $P(Z|X)$, para $Z$ variable latente y $X$ variable observada. Idealmente, si conociesemos todo,

$$
p(z|x)=\frac{p(x|z)p(z)}{p(x)},
$$

pero el problema usual es que el denominador $o(x)$ no tiene una forma cerrada. La inferencia variacional intenta encontrar otra distribución $q(z)$ que se asemeja "lo más posible" a $p(z|x)$. Para esto, utiliza a la divergencia de Kullback-Leibler como una medida de "cercanía" entre dos distribuciones. Por lo tanto en la inferencia variacional se intenta encontrar 

$$
\hat q := \text{argmin}_q KL(q(z)||p(z|x))
$$
y luego devuelve $\hat q (z)$ como la aproximación a la distribución posterior $p(z|x)$. Por ende, en la inferencia variacional se minimiza la KL, lo que acabamos de ver que es equivalente a maximizar el ELBO.

Conceptualmente, la inferencia variacional nos permite formular el problema de inferencia Bayesiana aproximada como un problema de optimización, y para esto, sabemos muy bien como utilizar PyTorch!

## Mezcla de densidades Gaussianas (MDN)

En una red de mezcla de densidades gaussianas (MDN del inglés, Mixture Density Network) se aproxima a la densidad de probabilidad de una variable $t$  condicionada a un vector $\vec{x}$, $p(t|\vec{x})$ {cite}`mdnMedium`. La idea fue propuesta por Christopher M. Bishop en su paper en 1994, {cite}`bishop1994`, en donde dice textualmente "Why settle for one guess when you can have a whole bunch of them" (cuya traducción podría ser algo como: por qué quedarse con una estimación si podemos generar un montón de estimaciones posibles).

En este tipo de modelo, la densidad de probabilidad condicional está representada por una combinación lineal de funciones de kernel, típicamente funciones Gaussianas (aunque podrían ser otras). Es decir,

$$
p(t|x) = \sum_{i=1}^m \alpha_i(x) \phi_i(t|x),
$$
en donde $\alpha_i(x)$ son los coeficientes que mezclan las densidades Gaussianas, determinando cuánto peso recibe cada componente $\phi_i(t|x)$ para este modelo. En el caso de la mezcla de densidades gaussianas, estas componentes se pueden escribir como

$$
\phi_i (t|x) = \frac{1}{(2\pi)^{c/2}\sigma_i(x)^c} \exp\left(-\frac{|t-\mu_i(x)|^2}{2\sigma_i(x)^2}\right),
$$
siendo unívocamente definidas por sus parámetros: media $\mu_i(x)$ y varianza $\sigma_i^2$. $c$ usualmente es 2.

De esta manera, podemos estimar la probabilidad condicional $p(t|\vec{x})$ siempre que podamos estimar $\alpha_i$, $\mu_i$ y $\phi_i$ para todo $i=1\cdots,N$, donde $N$ es el número de densidades gaussianas del modelo. Teniendo esto en cuenta, podríamos plantear una red neuronal tipo perceptrón múltiple, cuyas entradas corresponden al vector $\vec{x}$ y cuya salida corresponde a los valores de  $\alpha_i$, $\mu_i$ y $\phi_i$, a partir de los cuáles podemos plantear la mezcla gaussiana y estimar la probabilidad para $t$ dados los $\vec{x}$. Esta idea se resume en la siguiente figura:

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*UKuoYsGWis22cOV7KpLjVg.png)

Los coeficientes que mezclan las densidades gaussianas, $\alpha_i$, son cruciales ya que balancean la influencia de cada componente. Por ende, están normalizados, es decir que su suma es igual a 1 y esto se impone de manera sencilla mediante una función *Softmax* de la forma

$$
\alpha_i = \frac{\exp(z_i^\alpha)}{\sum_{j=1}^M\exp(z_j^\alpha)},
$$

en donde $z_i^\alpha$ es la salida de la red neuronal que va a estimar el valor de los coeficientes de mezcla $\alpha$. Lo mismo podemos hacer para el desvío estándar $\sigma_i$ que debe ser positivo. En vez de permitir que la red estime $\sigma_i$, hacemos que estime $z_i^\sigma$ y que luego obtengamos el desvío estándar de manera sencilla mediante 

$$
\sigma_i = \exp(z_j^{\sigma}).
$$

Vemos en estos casos como restringimos la posibilidad de $\sigma_i$ a valores positivos únicamente mediante una restricción dura o *hard constraint*.

El secreto está en cómo entrenar la red para que la probabilidad sea la que se observa en el conjunto de datos. Aquí es donde entra la maximización de la verosimilitud.

La verosimilitud de nuetros datos bajo el modelo MDN es el producto de las probabilidades asignadas a cada dato puntual. Es decir, es

$$
\mathcal L = \Pi_{q=1}^Q p(t^q | \vec{x}^q),
$$

en donde tenemos $Q$ datos. Es decir, acá estamos calculando la probabilidad que hayamos medido los datos que medimos dado nuestro modelo de mezcla gaussiana. Podríamos tratar de maximizar esta verosimilitud, pero por cuestiones numéricas conviene hacer dos cosas. Por un lado, como el logaritmo es una función monótonamente creciente, maximizar una función $f(x)$ es lo mismo que maximizar $\ln f(x)$, es decir el máximo sigue estando en el mismo lugar. Para simplificar la productoria, tomamos el logaritmo de esta función

$$
\ln \mathcal L = \ln \left\{ \Pi_{q=1}^Q p(t^q | \vec{x}^q) \right\}=\sum_{q=1}^Q \ln p(t^q | \vec{x}^q).
$$

Por otro lado, los algoritmos de optimización generalmente están acostumbrados a minimizar funciones de costo, por ende podemos tomar el negativo del logaritmo de la función anterior como nuestra función de costo. Si remplazamos la expresión para $\ln p(t^q | \vec{x}^q)$ nos queda que queremos optimizar

$$
-\ln \mathcal L = - \sum_{q=1}^Q \ln\left\{\sum_{i=1}^m \alpha_i(x) \phi_i(t|x)\right\}.
$$

### Ejemplo: Predicción de la sensación térmica

Supongamos que contamos con la siguiente base de datos

```{code-cell}ipython3
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/pandego/mdn-playground/refs/heads/main/data/01_raw/weather_dataset/weather_dataset_example.csv', delimiter=";")
print(data.shape)
data.head()
```
en donde contamos con 96453 mediciones de 7 variables: temperature (temperatura),	apparent_temperature (sensación térmica), 	humidity (humedad), 	wind_speed, (velocidad de viento) 	wind_bearing (dirección del viento),  	visibility (visibilidad),	pressure (presión). Queremos predecir la sensación térmica dada el resto de las variables (6 variables). Podríamos plantear un problema de regresión, pero también podríamos plantear una Mezcla de densidades gaussianas (MDN) para predecir la probabilidad de sensación térmica dadas las demás variables. Implementemos en python el entrenamiento de una MDN. Para esto, primero vamos a estandarizar los datos utilizando y luego dividir al conjunto de datos en entrenamiento (70%), validación (20%) y prueba (10%) utilizando las funcionalidades `StandardScaler` y `train_test_split` de `scikit-learn`. 

```{code-cell}ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

p_entrenamiento = 0.7
p_validacion = 0.2
p_prueba = 1 - p_entrenamiento - p_validacion

X = data.drop(columns=['apparent_temperature']).values
y = data['apparent_temperature'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=p_entrenamiento, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=p_validacion/(p_validacion + p_prueba), random_state=42)
```

Luego, para utilizar una base de datos customizada en `PyTorch`, debemos implementar una clase que hereda de `torch.utils.data.Dataset` de la siguiente manera:

```{code-cell}ipthon3
import torch
from torch.utils.data import Dataset

class DatasetTemperatura(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = DatasetTemperatura(X_train, y_train)
val_dataset = DatasetTemperatura(X_val, y_val)
test_dataset = DatasetTemperatura(X_test, y_test)

print(f'Tamaño del conjunto de entrenamiento: {len(train_dataset)}')
print(f'Tamaño del conjunto de validación: {len(val_dataset)}')
print(f'Tamaño del conjunto de prueba: {len(test_dataset)}')
```

Además, crearemos los `DataLoader`s que permiten luego hacer batches de 64 casos y aleatorización

```{code-cell}ipython3
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

Aquí viene la implementación de nuestro modelo MDN como una clase de `PyTorch`, siguiendo la lógica presentada al comienzo de esta sección:

```{code-cell}ipython3
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):
    def __init__(self, dim_entrada, dim_salida, num_ocultas, num_mezclas):
        super(MDN, self).__init__()
        self.capa_oculta = nn.Sequential(
            nn.Linear(dim_entrada, num_ocultas),
            nn.Tanh(),
            nn.Linear(num_ocultas, num_ocultas),
            nn.Tanh(),
        )
        self.z_alfa = nn.Linear(num_ocultas, num_mezclas)
        self.z_mu = nn.Linear(num_ocultas, num_mezclas * dim_salida)
        self.z_sigma = nn.Linear(num_ocultas, num_mezclas * dim_salida)
        self.num_mezclas = num_mezclas
        self.dim_salida = dim_salida

    def forward(self, x):
        capa_oculta = self.capa_oculta(x)
        alfa = F.softmax(self.z_alfa(capa_oculta), dim=1)
        sigma = torch.exp(self.z_sigma(capa_oculta)).view(-1, self.num_mezclas, self.dim_salida)
        mu = self.z_mu(capa_oculta).view(-1, self.num_mezclas, self.dim_salida)
        return alfa, sigma, mu
```

Para poder entrenar este modelo, hemos visto que debemos minimizar el negativo del logaritmo de la verosimilitud. Esta función de costo la podemos plantear de la siguiente manera:

```{code-cell}ipython3
def mdn_loss(alfa, sigma, mu, y, eps=1e-8):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # Ensure y has shape [batch, num_mezclas, dim_salida] to match `mu`/`sigma` for broadcasting
    y = y.view(-1, 1, 1).expand_as(mu)
    prob = torch.exp(m.log_prob(y))
    prob_weighted = prob * alfa.unsqueeze(2)
    prob_sum = torch.sum(prob_weighted, dim=1)
    nll = -torch.log(torch.clamp(prob_sum, min=eps))
    return torch.mean(nll)
```

Una vez definidos los datos, la función de costo y el modelo, podemos entrenar como hacemos usualmente. Podemos definir el número de épocas y si queremos entrenar utilizando una GPU.

```{code-cell}ipython3
num_epocas = 100
# Usar GPU si hay disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

modelo = MDN(dim_entrada=6, dim_salida=1, num_ocultas=50, num_mezclas=3)
modelo.to(device)

optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

for epoca in range(num_epocas):
    modelo.train()
    perdida_total = 0
    for x_batch, y_batch in train_dataloader:
        # Move batch to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizador.zero_grad()
        alfa, sigma, mu = modelo(x_batch)
        perdida = mdn_loss(alfa, sigma, mu, y_batch)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()
    perdida_media = perdida_total / len(train_dataloader)
    if (epoca + 1) % 10 == 0:
        print(f'Época {epoca+1}/{num_epocas}, Pérdida: {perdida_media:.4f}')

        modelo.eval()
        perdida_val_total = 0
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                alfa_val, sigma_val, mu_val = modelo(x_val)
                perdida_val = mdn_loss(alfa_val, sigma_val, mu_val, y_val)
                perdida_val_total += perdida_val.item()
        perdida_val_media = perdida_val_total / len(val_dataloader)
        print(f'  Pérdida de validación: {perdida_val_media:.4f}')
        modelo.train()

modelo.eval()
perdida_test_total = 0
with torch.no_grad():
    for x_test, y_test in test_dataloader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        alfa_test, sigma_test, mu_test = modelo(x_test)
        perdida_test = mdn_loss(alfa_test, sigma_test, mu_test, y_test)
        perdida_test_total += perdida_test.item()
perdida_test_media = perdida_test_total / len(test_dataloader)
print(f'Pérdida en el conjunto de prueba: {perdida_test_media:.4f}')
```

Vemos que la red aprende a minimizar el negativo de la log-verosimilitud y que funciona bien en el conjunto de datos de prueba que nunca fue utilizado durante el entrenamiento.

Finalmente, puede ser un poco difícil muestrear de un MDN ya que primero hay que elegir de cuál de las densidades gaussianas se muestreará (utilizando las constantes normalizadas como pesos) y luego hay que muestrear la densidad gaussiana con los valores de mu y sigma correspondiente al alfa seleccionado. La siguiente función realiza lo mencionado, teniendo en cuenta que $B$ es el tamaño del batch y $K$ la cantidad de Gaussianas.  


```{code-cell}ipython3
def mdn_sample(alfa, sigma, mu):
    """
    alfa: [B, K]
    sigma, mu: [B, K, 1]
    Devuelve una muestra y_hat: [B]
    """
    B, K = alfa.shape
    cat = torch.distributions.Categorical(probs=alfa)
    k_idx = cat.sample()  # [B]

    # Seleccionar mu y sigma de la componente elegida
    b_idx = torch.arange(B, device=alfa.device)
    mu_k = mu[b_idx, k_idx, 0]       # [B]
    sigma_k = sigma[b_idx, k_idx, 0] # [B]

    normal = torch.distributions.Normal(mu_k, sigma_k)
    y_hat = normal.sample()  # [B]
    return y_hat
```

Una vez implementada la función de muestreo de nuestra nueva probabilidad condicional estimada, podemos ver el resultado de la predicción con respecto a los valores medidos en el conjunto de prueba. Lo vamos a ver mediante un histograma de ambas distribuciones (medida y muestreada) como así también un gŕafico de valores de sensación térmica medidos vs. predichos.

```{code-cell}ipython3
import matplotlib.pyplot as plt
import numpy as np

modelo.eval()
y_true_all = []
y_pred_sample_all = []
y_pred_mean_all = []
with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        alfa, sigma, mu = modelo(x_batch)

        y_pred_s = mdn_sample(alfa, sigma, mu)

        y_pred_m = (alfa.unsqueeze(2) * mu).sum(dim=1)[:, 0]

        y_true_all.append(y_batch.detach().cpu().numpy())
        y_pred_sample_all.append(y_pred_s.detach().cpu().numpy())
        y_pred_mean_all.append(y_pred_m.detach().cpu().numpy())

y_true_all = np.concatenate(y_true_all).reshape(-1)
y_pred_sample_all = np.concatenate(y_pred_sample_all).reshape(-1)
y_pred_mean_all = np.concatenate(y_pred_mean_all).reshape(-1)

plt.figure()
plt.hist(y_true_all, bins=30, alpha=0.6, density=True, label="Medido")
plt.hist(y_pred_sample_all, bins=30, alpha=0.6, density=True, label="Predicción")
plt.xlabel("Sensación térmica (apparent_temperature, °C)")
plt.ylabel("Densidad")
plt.title("Distribución: Medido vs Predicho")
plt.legend()
plt.show()

plt.figure()
plt.scatter(y_true_all, y_pred_mean_all, alpha=0.6)
mn = min(y_true_all.min(), y_pred_mean_all.min())
mx = max(y_true_all.max(), y_pred_mean_all.max())
plt.plot([mn, mx], [mn, mx], c='red')

plt.xlabel("Medido (°C)")
plt.ylabel("Predicho (media de mezcla, °C)")
plt.title("Medido vs Predicho")
plt.show()
```

Vemos que a pesar de una subestimación de la sensación térmica por encima de 30$^\circ$C y por debajo de -20$^\circ$C, las predicciones concuerdan muy bien con los valores medidos y que la distribución de valores muestrada sigue la forma de la distribución medida.
## Autoencoder Variacional (VAE)

En su forma más simple un Autoencoder Variacional (VAE, del inglés *Variational Autoencoder*) es un modelo probabilístico que encuentra una representación latente de baja dimensionalidad de los datos {cite}`vaeBernstein`. Son utilizados para reducción de dimensionalidad como así también como modelos generativos. Un VAE es un tipo de **Autoencoder**, es decir un modelo que toma un vector de entrada $\vec{x}$, lo comprime a un espacio de menor dimensionalidad $\vec{z}$ y luego lo descomprime de nuevo en el intento de devolver nuevamente el vector $\vec{x}$, como se muestra en la figura a continuación:

![](https://raw.githubusercontent.com/mbernste/mbernste.github.io/master/images/autoencoder.png)

En este esquema se muestrá al vector de entrada $\vec{x}$ que se alimenta a una función, usualmente una red neuronal, $h_{\phi}(\vec{x})$ para generar un vector de salida de menor dimensionalidad $\vec z$ y luego otra función, también red neuronal, $f_\theta$ que toma al vector $\vec{z}$ y lo descomprime hacia una aproximación $\vec{x'}$ de $\vec{x}$. Las variables $\phi$ y $\theta$ denotan los parámetros de las redes neuronales.

En el caso de los VAE, estos son como autoencoders, pero en este caso son modelos probabilísticos en donde aprenden distribuciones.  Describen la probabilidad conjunta $p(\vec{x},\vec{z})$ para las muestras $\vec{x}$ y sus variables latentes asociadas $\vec{z}$.

El proceso generativo de los VAEs comienza con muestrear la variable latente $\vec{z} \in \mathbb R ^J$ de una distribución sencilla, como una distribución estándar normal. Acá $J$ es la dimensión del espacio latente que es menor a la dimensión del espacio vectorial inicial $D$. Es decir,

$$
\vec{z} \sim N(\mathbf 0, \mathbf{I})
$$


## Red generativa adversa (GAN)

## Normalizadores de flujo (NF)

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
