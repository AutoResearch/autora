# Search Space

DARTS uses a search space of operations to find the best model. The search space is defined by the set of operations that can be applied in each computation step of the model. These operations are also referred to as *primitives*. We can select from the following space of primitives:

- **zero**: The output of the computation $x_j$ is not dependent on its input $x_i$.
- **add**: The output of the computation $x_j$ amounts to its input $x_i$.
- **subtract**: The output of the computation $x_j$ amounts to $-x_i$.
- **mult**: The output of the computation $x_j$ is its input $x_i$ multiplied by some constant $a$.
- **linear**: The output of the computation $x_j$ is linearly dependent on its input $x_i$: $x_j = a * x_i + b$.
- **relu**: The output of the computation $x_j$ is a rectified linear function of its input $x_i$: $x_j = \max(0, x_i)$.
- **exp**: The output of the computation $x_j$ is exponentially dependent on its input $x_i$: $x_j = \exp(x_i)$.
- **logistic**: The output of the computation $x_j$ is a logistic function of its input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **sin**: The output of the computation $x_j$ is the sine function of its input $x_i$: $x_j = \sin(x_i)$.
- **cos**: The output of the computation $x_j$ is the cosine function of its input $x_i$: $x_j = \cos(x_i)$.
- **tanh**: The output of the computation $x_j$ is the hyperbolic tangent function of its input $x_i$: $x_j = \tanh(x_i)$.

Some of the primitives above may also be preceded by a linear transformation, allowing for more degrees of freedom in the search space:

- **linear_relu**: The output of the computation $x_j$ is a rectified linear function of its *linearly transformed* input $x_i$: $x_j = \max(0, (a * x_i + b)$.
- **linear_exp**: The output of the computation $x_j$ is exponentially dependent on its *linearly transformed* input $x_i$: $x_j = \exp(a * x_i + b)$.
- **linear_logistic**: The output of the computation $x_j$ is a logistic function of its *linearly transformed* input $x_i$: $x_j = \frac{1}{1 + \exp(-b * (a * x_i + b))}$.
- **linear_sin**: The output of the computation $x_j$ the sine function of its *linearly transformed* input $x_i$: $x_j = a * \sin(a * x_i + b)$.
- **linear_cos**: The output of the computation $x_j$ the cosine function of its *linearly transformed* input $x_i$: $x_j = a * \cos(a * x_i + b)$.
- **linear_tanh**: The output of the computation $x_j$ the hyperbolic tangent function of its *linearly transformed* input $x_i$: $x_j = a * \tanh(a * x_i + b)$.

Note that the following functions are available but currently not identifiable by DARTS (<font color="red">please use the following functions with caution</font>):

- **reciprocal**: The output of the computation $x_j$ is the multiplicative inverse of its input $x_i$: $x_j = \frac{1}{x_i}$.
- **ln**: The output of the computation $x_j$ is the natural logarithm of its input $x_i$: $x_j = \ln(x_i)$.
- **softplus**: The output of the computation $x_j$ is a softplus function of its input $x_i$: $x_j = \log(1 + \exp(a * x_i)) / a$.
- **softminus**: The output of the computation $x_j$ is a softminus function of its input $x_i$: $x_j = x_j - \log(1 + \exp(a * x_i)) / a$.

## Example

The following example sets up a search space with the following primitives:

- zero operation
- addition
- multiplication
- sigmoid operation

```python
primitives = [
    "zero",
    "add",
    "mult",
    "linear_exp",
]
```

We can then pass these primitives directly to the DARTS regressor:

```python
from autora.skl.darts import DARTSRegressor

darts_estimator = DARTSRegressor(
    primitives=primitives
)
```
