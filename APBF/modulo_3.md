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

El principio fundamental de las PINNs consiste en representar la solución de un problema físico mediante una red neuronal y definir una función de pérdida que penaliza no sólo el error respecto a los datos observadors, sino también la falta de cumplimiento de las ecucaciones que gobiernan la física del problema. Las ecuaciones diferenciales en derivadas parciales (PDE) vienen escritas en forma de derivadas parciales de la solución $u$, que puede depender de varias variables, como la posición, velocidad, tiempo, etc, que abreviaremos $u(\vec{x},t)$. Podemos decir que dada una PDE para $u(\vec{x},t)$ con evolución temporal, la podemos expresar en términos de una función $\mathcal{F}$ de sus derivadas parciales 

$$
u_{t} = \mathcal{F}(u_x, u_{xx}, \dots, u_{xx\dots x}),
$$
en donde el subíndice $_x$ indica la derivada parcial con respecto a las dimensiones espaciales (que podría incluir derivadas respecto a diferentes direcciones) y el subíndice $_t$ la variación temporal.

En el caso más sencillo de PINN, la red neuronal $q(\vec{x},t)$ aproxima la solución real del problema $u(\vec{x},t)$ a partir de datos observados. Gracias a la diferenciación automática vista en el capítulo anterior, resulta posible calcular las derivadas parciales de la red neuronal $q$ con respecto a sus variables de entrada $\vec{x},t$, es decir, resulta posible calcular el gradiente $\nabla_{\vec{x},t}{q(\vec{x},t)}$. Este gran hecho permite evaluar residuos de ecuaciones diferenciales ordinarias o parciales sin necesidad de discretizaciones como mallas o métodos de elementos finitos. Este enfoque resulta particularmente atractivo en escenarios en donde los datos son escasos, ruidosos o incompletos, pero se dispone de modelos físicos bien establecidos.

Las PINNs han demostrado utilidad en una amplia variedad de aplicaciones, incluyendo dinámica de fluídos, transferencia de calor, mecánica de sólidos, electromagnetismo, e incluse en sistemas biológicos. Éstas ofrecen una formulación unificada para problemas directos e inversos, permitiendo tanto la estimación de campos físicos como la identificación de parámetros desconocidos. 


### Función de costo física

Dada solución $u$, podemos computar el residuo $R$ como

$$
R  = u_t - \mathcal{F}(u_x, u_{xx}, \dots, u_{xx\dots x}) = 0,
$$
que naturalmente debe ser cero para la solución $u$. Si ahora planteamos la misma ecuación para una red neuronal $q$ no entrenada, resulta altamente probable que el valor no sólo no sea igual a cero, sino que sea bien distinto a cero. Si quisieramos que la red neuronal aproxime a la solución de la ecuación diferencial, entonces deberíamos exigir que el residuo $R$ para $q$ sea próximo o igual a cero. Esto se alinea muy bien con la manera de entrenar redes neuronales que hemos visto anteriormente, la función de costo. Podemos entrenar para minimizar este residuo en combinación con los términos de costo de aprendizaje automático tradicional, como MSE, MAE, etc. anteriormente vistos. Más aún, a medida que vamos aproximando la solución $u$ mediante $q$, podemos evaluar a la aproximación en puntos específicos ($\vec{x_0},\vec{x_n}$) en donde queremos que la solución cumpla con determinadas condiciones (como condiciones de contorno, o condiciones iniciales) y podemos comparar con la solución $u(\vec{x_i},t_i)=y_i$ para generar más términos de costo de la forma $q(\vec{x_i},t_i)-y_i$ que queremos minimizar. De esta forma nuestra función de costo objetivo de entrenamiento se puede escribir como

$$
\text{arg min}_{\theta} \sum_i \alpha_0 (q(\vec{x_i},t_i)-y_i)^2 + \alpha_1 R(x_i)
$$
en donde $\alpha_{0,1}$ denotan hiperparámetros que escalean la contribución del término supervisado y del residual respectivamente. Estos serían los términos de costo física descriptos, pero podría haber terminos de costo adicionales con sus factores de escala correspondientes.

Entendamos la ecuación anterior. El primer término es un término convencional, una función de costo L2. Si optimizacemos éste término solamente, la red neuronal aproximaría a las muestras de entrenamiento correctamente, pero podría promediar múltiples modos en las soluciones, funcionando erróneamente en regiones entre los puntos muestreados. Si, en cambio, optimizamos sólamente el segundo término (el residual físico), puede que la red neuronal satisfaga localmente la PDE pero que tenga dificultades encontrando una solución que ajuste globalmente. Esto puede suceder debido a que pueden haber muchas soluciones que satisfagan el mismo residual. Cuando se optimizan ambos términos en simultáneo, la red aprende a aproximar una solución específica a los datos de entrenamiento mientras que captura el conocimiento subyacente en la PDE. 

Notar que no tenemos ninguna garantía que el término residual alcance cero durante el entrenamiento. Es decir que este tratamiento propuesto impone restricciones suaves (o *soft constraints*), sin ninguna garantía del cumplimiento del constraint, sino más bien de aproximarse hacia ello. 

Planteando el problema de esta manera, estamos pensando en una representación neuronal de campo que llamaremos campo neuronal. Es decir, nuestra red neuronal $q$ está siendo optimizada de manera tal de satifacer $R=0$. Por lo tanto $q=q(\vec{x},\theta)$, donde elegimos los parámetros $\theta$ de la red de manera tal que $q\approx u$ lo más posible. A esto se le suele llamar red neuronal basada en la física (PINN, del inglés physics informed nueral network) y el artículo {cite}`raissi2019physics` sirve de excelente guía para entender cómo se utilizan en problemas inversos y hacia adelante.

### Ejemplo: Optimización de la ecuación de Burger con una PINN

Consideremos la tarea de reconstrucción como un problema inverso. Vamos a utilizar la ecuación de Burger 

$$
\frac{\partial u}{\partial t} + u \nabla u = \nu \nabla \cdot \nabla u,
$$
una ecuación simple pero a la vez no lineal en 1 dimensión. Supongamos que contamos con una serie de observaciones a tiempo $t=0.5$. La solución debe cumplir con el residual de la formulación de la ecuación de Burger como a su vez coincidir con las observaciones en los puntos medidos. A su vez, imponemos la condición de contorno de Dirichlet $u=0$ en los bordes del dominio computacional y definimos la solución en el intervalo de tiempo $t\in [0,1]$.


```{code-cell} ipython3
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Network
# ----------------------------
class Network(nn.Module):
    def __init__(self, hidden=20, depth=8):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if x.shape != t.shape:
            raise ValueError(f"x and t must have same shape, got {x.shape} vs {t.shape}")
        y = torch.stack([x, t], dim=-1)      # (..., 2)
        y_flat = y.reshape(-1, 2)            # (M, 2)
        out_flat = self.net(y_flat)          # (M, 1)
        out = out_flat.reshape(*x.shape, 1) # (..., 1)
        return out


_model = Network().to(device)

def network(x, t):
    return _model(x, t)

# ----------------------------
# Boundary / sampling utilities
# ----------------------------
def boundary_tx(N, device=None, dtype=torch.float32):
    x = torch.linspace(-1, 1, 128, device=device, dtype=dtype)
    u = torch.tensor( [0.008612174447657694, 0.02584669669548606, 0.043136357266407785, 0.060491074685516746, 0.07793926183951633, 0.0954779141740818, 0.11311894389663882, 0.1308497114054023, 0.14867023658641343, 0.1665634396808965, 0.18452263429574314, 0.20253084411376132, 0.22057828799835133, 0.23865132431365316, 0.25673879161339097, 0.27483167307082423, 0.2929182325574904, 0.3109944766354339, 0.3290477753208284, 0.34707880794585116, 0.36507311960102307, 0.38303584302507954, 0.40094962955534186, 0.4188235294008765, 0.4366357052408043, 0.45439856841363885, 0.4720845505219581, 0.4897081943759776, 0.5072391070000235, 0.5247011051514834, 0.542067187709797, 0.5593576751669057, 0.5765465453632126, 0.5936507311857876, 0.6106452944663003, 0.6275435911624945, 0.6443221318186165, 0.6609900633731869, 0.67752574922899, 0.6939334022562877, 0.7101938106059631, 0.7263049537163667, 0.7422506131457406, 0.7580207366534812, 0.7736033721649875, 0.7889776974379873, 0.8041371279965555, 0.8190465276590387, 0.8337064887158392, 0.8480617965162781, 0.8621229412131242, 0.8758057344502199, 0.8891341984763013, 0.9019806505391214, 0.9143881632159129, 0.9261597966464793, 0.9373647624856912, 0.9476871303793314, 0.9572273019669029, 0.9654367940878237, 0.9724097482283165, 0.9767381835635638, 0.9669484658390122, 0.659083299684951, -0.659083180712816, -0.9669485121167052, -0.9767382069792288, -0.9724097635533602, -0.9654367970450167, -0.9572273263645859, -0.9476871280825523, -0.9373647681120841, -0.9261598056102645, -0.9143881718456056, -0.9019807055316369, -0.8891341634240081, -0.8758057205293912, -0.8621229450911845, -0.8480618138204272, -0.833706571569058, -0.8190466131476127, -0.8041372124868691, -0.7889777195422356, -0.7736033858767385, -0.758020740007683, -0.7422507481169578, -0.7263049162371344, -0.7101938950789042, -0.6939334061553678, -0.677525822052029, -0.6609901538934517, -0.6443222327338847, -0.6275436932970322, -0.6106454472814152, -0.5936507836778451, -0.5765466491708988, -0.5593578078967361, -0.5420672759411125, -0.5247011730988912, -0.5072391580614087, -0.4897082914472909, -0.47208460952428394, -0.4543985995006753, -0.4366355580500639, -0.41882350871539187, -0.40094955631843376, -0.38303594105786365, -0.36507302109186685, -0.3470786936847069, -0.3290476440540586, -0.31099441589505206, -0.2929180880304103, -0.27483158663081614, -0.2567388003912687, -0.2386513127155433, -0.22057831776499126, -0.20253089403524566, -0.18452269630486776, -0.1665634500729787, -0.14867027528284874, -0.13084990929476334, -0.1131191325854089, -0.09547794429803691, -0.07793928430794522, -0.06049114408297565, -0.0431364527809777, -0.025846763281087953, -0.00861212501518312], device=device, dtype=dtype)
    t = torch.full_like(x, 0.5)
    perm = torch.randperm(128, device=device)
    return x[perm][:N], t[perm][:N], u[perm][:N]

def open_boundary(N, device=None, dtype=torch.float32):
    t = torch.rand(N, device=device, dtype=dtype)
    half = N // 2
    x = torch.cat(
        [torch.ones(half, device=device, dtype=dtype),
         -torch.ones(half, device=device, dtype=dtype)],
        dim=0
    )
    u = torch.zeros(N, device=device, dtype=dtype)
    return x, t, u

# ----------------------------
# Autograd helpers
# ----------------------------
def gradients(y, x):
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def burgers_residual(u, x, t):
    u_t  = gradients(u, t)
    u_x  = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u * u_x - (0.01 / np.pi) * u_xx

# ----------------------------
# Grid for visualization
# ----------------------------
N = 128
grids_xt = np.meshgrid(
    np.linspace(-1, 1, N),
    np.linspace(0, 1, 33),
    indexing="ij"
)
grid_x = torch.tensor(grids_xt[0], dtype=torch.float32, device=device)
grid_t = torch.tensor(grids_xt[1], dtype=torch.float32, device=device)

with torch.no_grad():
    grid_u = network(grid_x, grid_t).unsqueeze(0)

# ----------------------------
# Visualization helper
# ----------------------------
# ----------------------------
# Visualization helper
# ----------------------------
def show_state(a, title):
    U = a[0, :, :, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 5))
    im = plt.imshow(
    U,
    origin="upper",
    aspect="auto",
    cmap="inferno",
    extent=[0, 1, -1, 1]
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_title(title)
    plt.show()

print("Randomly initialized network state:")
show_state(grid_u, "Uninitialized NN")

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)

# ----------------------------
# Boundary data
# ----------------------------
N_SAMPLE_POINTS_BND = 100
x_t0, t_t0, u_t0 = boundary_tx(N_SAMPLE_POINTS_BND, device=device)
x_ob, t_ob, u_ob = open_boundary(N_SAMPLE_POINTS_BND, device=device)

x_bc = torch.cat([x_t0, x_ob], dim=0)
t_bc = torch.cat([t_t0, t_ob], dim=0)
u_bc = torch.cat([u_t0, u_ob], dim=0)

# ----------------------------
# Training loop
# ----------------------------
N_SAMPLE_POINTS_INNER = 1000
ITERS = 10_000
ph_factor = 1.0

start = time.time()

for step in range(ITERS + 1):
    optimizer.zero_grad()

    # Boundary loss
    u_pred_bc = network(x_bc, t_bc).reshape(-1)
    loss_u = torch.mean((u_pred_bc - u_bc.reshape(-1)) ** 2)

    # Physics loss (resample each iter)
    x_ph = (2.0 * torch.rand(N_SAMPLE_POINTS_INNER, device=device) - 1.0).requires_grad_(True)
    t_ph = torch.rand(N_SAMPLE_POINTS_INNER, device=device).requires_grad_(True)

    u_pred_ph = network(x_ph, t_ph).reshape(-1)
    residual = burgers_residual(u_pred_ph, x_ph, t_ph)
    loss_ph = torch.mean(residual ** 2)

    loss = loss_u + ph_factor * loss_ph
    loss.backward()
    optimizer.step()

    if step < 3 or step % 1000 == 0:
        with torch.no_grad():
            grad_norm = torch.sqrt(
                sum((p.grad**2).sum() for p in _model.parameters() if p.grad is not None)
            )
        print(
            f"Step {step:6d} | "
            f"Total: {loss.item():.6e} | "
            f"Boundary: {loss_u.item():.6e} | "
            f"Physics: {loss_ph.item():.6e} | "
            f"|grad|: {grad_norm.item():.3e}"
        )
end = time.time()
print(f"Runtime {end - start:.2f}s")

# ----------------------------
# After training visualization
# ----------------------------
_model.eval()
with torch.no_grad():
    grid_u = network(grid_x, grid_t).unsqueeze(0)

show_state(grid_u, "After Training")

```
<!-- ## Diferenciación automática para ecuaciones diferenciales parciales con constraints

Las redes neuronales soportan inherentemente el cálculo de las derivadas con respecto al vector de entrada. La derivada $\partial f / \partial \theta$ es un ingrediente fundamental para aprender por el descenso por el gradiente. 

Si el vector de entrada es $\vec{x}$ -->

## Física diferenciable

En esta sección vamos a explorar la posibilidad de incorporar simulaciones numéricas diferenciables al proceso de aprendizaje, lo cuál abreviaremos como física diferenciable (FD). 


El objetivo central de esta metodología es aprovechar los algorítmos numéricos existentes para resolver ecuaciones diferenciales para mejorar los sistemas de inteligencia artificial. Para esto, resulta necesario equipar a los sistemas de IA con la funcionalidad de calcular los gradientes con respecto a sus entradas. Una vez que hacemos esto para todos los opoeradores de una simulación, podemos utilizar la diferenciación automática {cite}`baydin2018automatic` a nuestro favor junto con la retropropagación para permitir que la información del gradiente fluya del simulador a la red neuronal y viceversa. 

En contraste con las PINNs propuestas anteriormente, esto permite manejar espacios de soluciones más complejos, sin necesidad de aprender para un problema inverso específico como hicimos anteriormente. La física diferenciable permite entrenar redes neuronales que aprender a aproximar soluciones a un conjunto mayor de problemas inversos de manera más eficiente. 


### Operadores diferenciables

En FD trabajamos encima de solvers numéricos existentes. Depende fuertemente en los algoritmos computacionales ya disponibles en cada área de la física. Para comenzar, necesitamos una formulación contínua como modelo para el efecto física que queremos simular. 

Asumimos que tenemos una formulación contínua $\mathcal{P}^*(\vec{u}, \nu)$ de la cantidad física de interés $\vec{u}(\vec{u},t) : \mathbb{R}^d\times \mathbb{R}^+ \rightarrow \mathbb{R}^d$, con parámetros de modelo $\nu$ (por ejemplo, constante de difusión, viscocidad, conductividad, etc.). Las componentes de $u$ estarán denotadas por un  un sub-índice $i$ ($\vec{u}=(u_1,\dots ,u_d)^T$). Típicamente estamos interesados en la evolución temporal de \vec{u}. Discretizamos en intervalos de tiempo $\Delta t$ y esto resulta en una formulación $\mathcal{P}(\vec{u},\nu)$. El estado a tiempo $t+\Delta t$ se computa mediante la secuencia de operadores $\vec{\mathcal{P}_1, \mathcal{P}_2, \dots, \mathcal{P}_m}$ de tal manera que $\vec{u}(t+\Delta t) = \mathcal{P}_m \circ \dots \circ \mathcal{P}_2 \circ \mathcal{P}_1(\vec{u}(t), \nu)$, donde $\circ$ denota la composición de funcioines, i.e. $f\circ g(x) = f(g(x))$. 

Para incorporar este solver numérico al proces de aprendizaje profundo, necesitamos contar con el gradiente de cada operador $\mathcal{P}_i$ con respecto a sus inputs, i.e. $\partial \mathcal{P}_i/\partial \vec{u}$. Notar que no necesitamos siempre las derivadas con respecto a todos los parámetros (por ejemplo tal vez no queremos optimizar con respecto de $\nu$), con lo cual se pueden omitir ciertas derivadas. En lo que sigue asumimos que $\nu$ es un parámetro del modelo, pero que no va a ser una de las salidas de nuestra red neuronal, para evitar pasar $\partial \mathcal{P}_i/\partial \vec{\nu}$ a nuestro solver numérico.

### Jacobianos

Como típicamente $\vec{u}$ es un vector, $\partial \mathcal{P}_i/\partial \vec{u}$ denota una matriz Jacobiana $J$ en vez de un único valor, i.e. 

$$
\frac{\partial \mathcal{P}_i}{\partial \vec{u}} = \nabla_{\vec{u}}\mathcal{P}_i = \begin{bmatrix} 
    \partial \mathcal P_{i,1} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,1} / \partial u_{d} 
    \\
    \vdots & \ & \ 
    \\
    \partial \mathcal P_{i,d} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,d} / \partial u_{d} 
    \end{bmatrix},
$$
en donde $d$ denota el número de componentes de $\vec{u}$. En este caso, como $\mathcal{P}$ mapea un valor de $\vec{u}$ a otro valor de de $\vec{u}$, el jacobiano es cuadrado, pero podría ser que este no sea el caso sin traer ningún tipo de problema a la metodología propuesta. 

Ahora utilizaremos el modo reverso de diferenciación automática y nos centraremos en computar un producto vectorial de matrices entre la transpuesta del jacobiano y un vector $\vec{a}$, .i.e. $\left(\frac{\partial \mathcal{P}_i}{\partial \vec{u}}\right)^T \vec{a}$. Si tuviesemos que construir y almacenar toda matriz jacobiana que necesitamos durante el entrenamiento causaría mucho uso de memoria y relentizaría el proceso de entrenamiento innecesariamente. En vez de eso, en la retropropagación, podemos computar productos con el jacobiano más rápidos porque siempre tenemos una función escalar de costo al final de la cadena. 


eniendo en cuenta esta formulación, necesitamos resolver la derivada de la función de costo escalar $\mathcal l$, lo cual es equivalente a considerar las derivadas de la cadena de funciones compuestas $\mathcal{P}_i$ evaluadas en un estado actual dado $\vec{u}^n$ mediante la regla de la cadena. A modo de ejemplo, para el caso de dos operadores


$$
    \frac{\partial \mathcal l}{ \partial \mathbf{u}}  = \frac{ \partial (\mathcal P_2 \circ \mathcal P_1) }{ \partial \mathbf{u} } \Big|_{\mathbf{u}^n}
    = 
    \frac{ \partial \mathcal P_2 }{ \partial \mathcal P_1 } \big|_{\mathcal P_1(\mathbf{u}^n)}
    \ 
    \frac{ \partial \mathcal P_1 }{ \partial \mathbf{u} } \big|_{\mathbf{u}^n} \ ,
$$
o cual corresponde a la versión vectorial de la regla de la cadena clásica y se extiende de forma directa al caso de una composición de más de dos operadores $i>2$.

Las derivadas de $\mathcal{P}_1$ y $\mathcal{P}_2$ siguen siendo jacobianos, pero dado que la función de costo $\mathcal l$ es escalar, el gradiente de $\mathcal l$ con respecto al último operador de la cadena es un vector. En el modo reverso de diferenciación, se inicia la propagación con este gradiente y se calculan sucesivamente los productos Jacobiano propagando la información de sensibilidad hacia el estado inicial \vec{u}.


De esta manera, una vez que podemos calcular los productos de los Jacobianos de los operadores de nuestro simulador, podemos integrarlos dentro del workflow de aprendizaje profundo. 



```{bibliography}
:style: unsrt
:filter: docname in docnames
```