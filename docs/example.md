# 

## Example

Let's generate a simple data set with two features $x_1, x_2 \in [0, 1]$ and a target $y$. We will use the following generative model: 
$y = 2 x_1 - e^{(5 x_2)}$

```python
import numpy as np

x_1 = np.linspace(0, 1, num=10)
x_2 = np.linspace(0, 1, num=10)
X = np.array(np.meshgrid(x_1, x_2)).T.reshape(-1,2)

y = 2 * X[:,0] + np.exp(5 * X[:,1])
```

Now let us choose a prior over the primitives. In this case, we will use priors determined by Guimerà et al (2020).

```python
prior = "Guimera2020"
```

## Set up the BMS Regresssor

We will use the BMS Regresssor to predict the outcomes. There are a number of parameters that determine how the architecture search is performed. The most important ones are listed below:

- **`epochs`**: The number of epochs to run BMS. This corresponds to the total number of equation mutations - one mcmc step for each parallel-tempered equation and one tree swap between a pair of parallel-tempered equations
- **`prior_par`**: A dictionary of priors for each operation. The keys correspond to operations and the values correspond to a measure of the prior probability of that operation occurring. The model comes with a default.  
- **`ts`**: A list of temperature values. The machine scientist creates an equation tree for each temperature. Higher temperature trees are harder to fit, and thus they help to avoid overfitting the model.


Let's set up the BMS regressor with default parameters.

```python
from autora.skl.bms import BMSRegressor

bms_estimator = BMSRegressor()
```

Now we have everything to fit and verify the model.

```python
bms_estimator.fit(X,y)
bms_estimator.predict(X)
```

## Troubleshooting

We can troubleshoot the model by playing with a few parameters:
- Increasing the number of epochs. The original paper recommends 1500-3000 epochs for reliable fitting. The default is set to 1500.
- Using custom priors, more relevant to the data. The default priors are over equations nonspecific to scientific domain.
- Increasing the range of temperature values to escape local minima.
- reduce the differences between parallel temperatures to escape local minima.
