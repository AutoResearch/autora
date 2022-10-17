# Bayesian Machine Scientist

## Search space

BMS uses a search space of operations, along with parameters to find the best model. The search space is very rugged, and so multiple equations at parallel temperatures are considered. The search space is defined by the set of operations that can be applied in each computation step of the model. These operations are also referred to as *primitives*. We can select from the following space of primitives:

- **\textit{constant}**: The output of the computation $x_j$ is a constant parameter value $a$ where $a$ is a fitted float value.
- **\+**: The output of the computation $x_j$ is the sum over its two inputs $x_i,x_{ii}$.
- **\-**: The output of the computation $x_j$ is the respective difference between its inputs $x_i,x_{ii}$.
- **\***: The output of the computation $x_j$ is the product over its two inputs $x_i,x_{ii}$.
- **\/**: The output of the computation $x_j$ is the resective quotient between its inputs $x_i,x_{ii}$.
- **relu**: The output of the computation $x_j$ is a rectified linear function of its input $x_i$: $x_j = \max(0, x_i)$.
- **exp**: The output of the computation $x_j$ is exponentially dependent on its input $x_i$: $x_j = \exp(x_i)$.
- **sig**: The output of the computation $x_j$ is a logistic function of its input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **fac**: The output of the computation $x_j$ is the generalized factorial function of its input $x_i$: $x_j = \Gamma(1 + x_i)$.
- **sin**: The output of the computation $x_j$ is the sinus function of its input $x_i$: $x_j = \sin(x_i)$.
- **sinh**: The output of the computation $x_j$ is the sinus hyperbolicus of its input $x_i$: $x_j = \sinh(x_i)$.
- **cos**: The output of the computation $x_j$ is the cosinus function of its input $x_i$: $x_j = \cos(x_i)$.
- **cosh**: The output of the computation $x_j$ is the cosinus hyperbolicus of its input $x_i$: $x_j = \cosh(x_i)$.
- **tan**: The output of the computation $x_j$ is the tangens of its input $x_i$: $x_j = \tan(x_i)$.
- **tanh**: The output of the computation $x_j$ is the tangens hyperbolicus of its input $x_i$: $x_j = \tanh(x_i)$.


# Example

How we search the search space is determined by the priors. The following example sets up the BMS Regressor with priors over psychology wiki equations. This will optimize the model to search for equations most prevalent in the psychology. 

```python
from autora.skl.bms import BMSRegressor

bms_estimator = BMSRegressor(
    prior="Psychology2022"
)
```
