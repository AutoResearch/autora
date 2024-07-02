# Search Space

The following are built-in operators which constitute the search space:

- **\+**: The output of the computation $x_j$ is the sum over its inputs $x_i, x_{ii}$: $x_j = x_i + x_{ii}$.
- **\-**: The output of the computation $x_j$ is the respective difference between its inputs $x_i, x_{ii}$: $x_j = x_i - x_{ii}$.
- __\*__: The output of the computation $x_j$ is the product over its two inputs $x_i, x_{ii}$: $x_j = x_i * x_{ii}$.
- **exp**: The output of the computation $x_j$ is the natural exponential function applied to its input $x_i$: $x_j = \exp(x_i)$.
- **pow2**: The output of the computation $x_j$ is the square function applied to its input $x_i$: $x_j$ = $x_i^2$.
- **pow3**: The output of the computation $x_j$ is the cube function applied to its input $x_i$: $x_j$ = $x_i^3$.
- **sin**: The output of the computation $x_j$ is the sine function applied to its input $x_i$: $x_j = \sin(x_i)$.
- **cos**: The output of the computation $x_j$ is the cosine function applied to its input $x_i$: $x_j = \cos(x_i)$.
- **ln**: The output of the computation $x_j$ is the linear transformation applied to its input $x_i$: $x_j = a * x_i + b$, where $a$ and $b$ are slope and intercept parameters.

In BSR, a new operator can be added in two steps. First, define an operator as a function, as demonstrated in `operations.py`. Second, add the name of the operator and its prior information to the dictionaries in `__get_prior()` within `prior.py`.
