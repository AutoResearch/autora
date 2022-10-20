# Bayesian Machine Scientist

## Search space

BMS uses a search space of operations, along with parameters to find the best model. Since the search space is very rugged, BMS considers multiple equation trees in parallel, with each tree governed by a different temperature parameter. The search space is defined by the set of operations that can be applied in each computation step of the model. These operations are also referred to as *primitives*. We can select from the following space of primitives:

- **$\textit{constant}$**: The output of the computation $x_j$ is a constant parameter value $a$ where $a$ is a fitted float value.
- **\+**: The output of the computation $x_j$ is the sum over its two inputs $x_i, x_{ii}$: $x_j = x_i \+ x_{ii}$.
- **\-**: The output of the computation $x_j$ is the respective difference between its inputs $x_i, x_{ii}$: $x_j = x_i \- x_{ii}$.
- **\***: The output of the computation $x_j$ is the product over its two inputs $x_i, x_{ii}$: $x_j = x_i \* x_{ii}$.
- **\/**: The output of the computation $x_j$ is the respective quotient between its inputs $x_i, x_{ii}$: $x_j = x_i \/ x_{ii}$.
- **abs**: The output of the computation $x_j$ is the absolute value of its input $x_i$: $x_j = |(x_i)|$.
- **relu**: The output of the computation $x_j$ is a rectified linear function applied to its input $x_i$: $x_j = \max(0, x_i)$.
- **exp**: The output of the computation $x_j$ is the natural exponential function applied to its input $x_i$: $x_j = \exp(x_i)$.
- **log**: The output of the computation $x_j$ is the natural logarithm function applied to its input $x_i$: $x_j = \log(x_i)$.
- **sig**: The output of the computation $x_j$ is a logistic function applied to its input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **fac**: The output of the computation $x_j$ is the generalized factorial function applied to its input $x_i$: $x_j = \Gamma(1 + x_i)$.
- **sqrt**: The output of the computation $x_j$ is the square root function applied to its input $x_i$: $x_j = \sqrt(x_i)$.
- **pow2**: The output of the computation $x_j$ is the square function applied to its input $x_i$: $x_j = x_i^2 $.
- **pow3**: The output of the computation $x_j$ is the cube function applied to its input $x_i$: $x_j = x_i^3$.
- **sin**: The output of the computation $x_j$ is the sine function applied to its input $x_i$: $x_j = \sin(x_i)$.
- **sinh**: The output of the computation $x_j$ is the hyperbolic sine function applied to its input $x_i$: $x_j = \sinh(x_i)$.
- **cos**: The output of the computation $x_j$ is the cosine function applied to its input $x_i$: $x_j = \cos(x_i)$.
- **cosh**: The output of the computation $x_j$ is the hyperbolic cosine function applied to its input $x_i$: $x_j = \cosh(x_i)$.
- **tan**: The output of the computation $x_j$ is the tangent function applied to its input $x_i$: $x_j = \tan(x_i)$.
- **tanh**: The output of the computation $x_j$ is the hyperbolic tangent function applied to its input $x_i$: $x_j = \tanh(x_i)$.
- **\*\***: The output of the computation $x_j$ is the product over its two inputs $x_i,x_{ii}$: $x_j = x_i \*\* x_{ii}$.

# Example

How we search the search space is determined by the priors. The following example sets up the BMS Regressor with priors over equations found in Wikipedia pages that are tagged with the psychology category. This will optimize the model to search for equations most prevalent in the field of psychology. 

```python
from autora.skl.bms import BMSRegressor

bms_estimator = BMSRegressor(
    prior="Psychology2022"
)
```
