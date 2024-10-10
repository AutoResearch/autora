# Random Pooler

Creates combinations from lists of discrete values using random selection.

## Example


To illustrate the concept of a random pool of size 3, let's consider a situation where a certain condition is defined by two variables: $x_{1}$ and $x_{2}$. The variable $x_{1}$ can take on the values of 1, 2, or 3, while $x_{2}$ can take on the values of 4, 5, or 6.

| $x_{1}$ | $x_{2}$ |
|---------|---------|
| 1       | 4       |
| 2       | 5       |
| 3       | 6       |

This means that there are 9 possible combinations for these variables (3x3), from which a random pool of size 3 draws 3 combinations.

|    | 4     | 5     | 6   |
|----|-------|-------|-----|
| 1  | X     | (1,5) | X   |
| 2  | X     | X     | X   |
| 3  | (3,4) | (3,5) | X   |

### Example Code

```python

from autora.experimentalist.random import random_pool

pool = random_pool([1, 2, 3], [4, 5, 6], num_samples=3)
```
