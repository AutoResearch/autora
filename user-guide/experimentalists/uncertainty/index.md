# Uncertainty Experimentalist

The uncertainty experimentalist identifies experimental conditions $\vec{x}' \in X'$ with respect model uncertainty. Within the uncertainty experimentalist, there are three methods to determine uncertainty:

## Least Confident
$$
x^* = \text{argmax} \left( 1-P(\hat{y}|x) \right),
$$

where $\hat{y} = \text{argmax} P(y_i|x)$

## Margin

$$
x^* = \text{argmax} \left( P(\hat{y}_1|x) - P(\hat{y}_2|x) \right),
$$

where $\hat{y}_1$ and $\hat{y}_2$ are the first and second most probable class labels under the model, respectively.

## Entropy
$$ 
x^* = \text{argmax} \left( - \sum P(y_i|x)\text{log} P(y_i|x) \right)
$$

# Example Code

```
from autora.experimentalist.uncertainty import uncertainty_sample
from sklearn.linear_model import LogisticRegression
import numpy as np

#Meta-Setup
X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
y = (X**2).reshape(-1)
n = 5

#Theorists
lr_theorist = LogisticRegression()
lr_theorist.fit(X,y)

#Experimentalist
X_new = uncertainty_sample(X, lr_theorist, n, measure ="least_confident")
```
