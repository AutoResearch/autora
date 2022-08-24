# Differentiable Architecture Search

## Search space

DARTS uses a search space of operations to find the best model. The search space is defined by the set of operations that can be applied in each computation step of the model. These operations are also referred to as *primitives*. We can select from the following space of primitives:

- **zero**: The output of the computation $x_j$ is not dependent on its input $x_i$.
- **add**: The output of the computation $x_j$ is the sum of its input $x_i$ and some constant $a$.
- **mult**: The output of the computation $x_j$ is its input $x_i$ multiplied by some constant $a$.
- **linear**: The output of the computation $x_j$ is linear dependent on its input $x_i$: $x_j = a * x_i + b$.
- **exponential**: The output of the computation $x_j$ is exponentially dependent on its input $x_i$: $x_j = a * \exp(b * x_i)$.
- **sigmoid**: The output of the computation $x_j$ is a logistic function of its input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **lin_sigmoid**: The output of the computation $x_j$ is a logistic function of its *linearly transformed* input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **relu**: The output of the computation $x_j$ is a rectified linear function of its input $x_i$: $x_j = \max(0, x_i)$.
- **lin_relu**: The output of the computation $x_j$ is a rectified linear function of its *linearly transformed* input $x_i$: $x_j = \max(0, x_i)$.
- **softplus**: The output of the computation $x_j$ is a softplus function of its input $x_i$: $x_j = \log(1 + \exp(a * x_i)) / a$.
- **softminus**: The output of the computation $x_j$ is a softminus function of its input $x_i$: $x_j = x_j - \log(1 + \exp(a * x_i)) / a$.

# Example

The following example sets up a search space with the following primitives:
- zero operation
- addition
- multiplication
- sigmoid operation

```python

PRIMITIVES = [
        "zero",
        "add",
        "mult",
        "sigmoid",
    ],

```

We can then pass these primitives directly to the DARTS regressor:

```python
from autora.skl.darts import DARTSRegressor, ValueType

darts_estimator = DARTSRegressor(
    primitives=PRIMITIVES,
)
```
