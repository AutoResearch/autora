## Example

First me must import the necessary modules

```python
from autora.skl.darts import DARTSRegressor, ValueType
```

Let's generate a simple data set with two features $x_1, x_2 \in [0, 1]$ and a target $y$. We will use the following generative model: 
$$y = exp(x_1) - x_2$$

```python
import numpy as np

x_1 = np.linspace(0, 1, num=10)
x_2 = np.linspace(0, 1, num=10)
X = np.array(np.meshgrid(x_1, x_2)).T.reshape(-1,2)

y = 2 * X[:,0] + np.exp(5 * X[:,1])
```

Now let us define the search space, that is, the space of operations to consider when searching over the space of computation graphs.

```python

PRIMITIVES = [
    "none",
    "add",
    "subtract",
    'mult',
    "sigmoid",
    'exp',
    'relu',
]

```

## Set up the DARTS Regresssor

We will use the DARTS Regresssor to predict the outcomes. There are a number of parameters that determine how the architecture search is performed. The most important ones are listed below:

- **num_graph_nodes**: The number of latent variables used to represent the model.
- **arch_updates_per_epoch**: The number of architecture updates per training epoch. These updates affect the architecture weights $\alpha$ indicating the relative contribution of each operation for a given computation step.
- **param_updates_per_epoch**: The number of parameter updates per epoch. Once the architecture updates are complete, the parameters associated with each operation are updated.
- **max_epochs**: The maximum number of epochs to run DARTS.
- **output_type**: The type of output to produce. In our case, we treat the outcome as a real variable, i.e., ValueType.REAL.

Let's set up the DARTS regressor with some default parameters.

```python
from autora.skl.darts import DARTSRegressor, ValueType

darts_estimator = DARTSRegressor(
    arch_updates_per_epoch=1,
    param_updates_per_epoch=500,
    max_epochs=100,
    output_type=ValueType.REAL,
    num_graph_nodes=2,
)
```

Now we have everything to run differentiable architecture search and visualize the model resulting from the highest architecture weights. Note that the current model corresponds to the model with the highest architecture weights.

```python
darts_estimator.fit(X, y)
darts_estimator.visualize_model()
```
