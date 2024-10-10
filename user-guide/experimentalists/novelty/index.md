# Novelty Experimentalist

The novelty experimentalist identifies experimental conditions $\vec{x}' \in X'$ with respect to
a pairwise distance metric applied to existing experimental conditions $\vec{x} \in X$:

$$
\underset{\vec{x}'}{\arg\max}~f(d(\vec{x}, \vec{x}'))
$$

where $f$ is an integration function applied to all pairwise  distances.

## Example

For instance,
the integration function $f(x)=\min(x)$ and distance function $d(x, x')=|x-x'|$ identifies
condition $\vec{x}'$ with the greatest minimal Euclidean distance to all
existing conditions in $\vec{x} \in X$.

$$
\underset{\vec{x}}{\arg\max}~\min_i(\sum_{j=1}^n(x_{i,j} - x_{i,j}')^2)
$$

To illustrate this sampling strategy, consider the following four experimental conditions that
were already probed:


| $x_{i,0}$ | $x_{i,1}$ | $x_{i,2}$ |
|-----------|-----------|-----------|
| 0         | 0         | 0         |
| 1         | 0         | 0         |
| 0         | 1         | 0         |
| 0         | 0         | 1         |

Fruthermore, let's consider the following three candidate conditions $X'$:

| $x_{i,0}'$ | $x_{i,1}'$ | $x_{i,2}'$ |
|------------|------------|------------|
| 1          | 1          | 1          |
| 2          | 2          | 2          |
| 3          | 3          | 3          |


If the novelty experimentalist is tasked to identify two novel conditions, it will select
the last two candidate conditions $x'_{1,j}$ and $x'_{2,j}$ because they have the greatest
minimal distance to all existing conditions $x_{i,j}$:

### Example Code
```python
import numpy as np
from autora.experimentalist.novelty import novelty_sample, novelty_score_sample

# Specify X and X'
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_prime = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# Here, we choose to identify two novel conditions
n = 2
X_sampled = novelty_sample(conditions=X_prime, reference_conditions=X, num_samples=n)

# We may also obtain samples along with their z-scored novelty scores  
(X_sampled, scores) = novelty_score_sample(conditions=X_prime, reference_conditions=X, num_samples=n)
```




