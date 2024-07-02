# Grid Pooler

Creates exhaustive pool from discrete values using a Cartesian product of sets.

## Example

To illustrate the concept of an exhaustive pool, let's consider a situation where a certain condition is defined by two variables: $x_{1}$ and $x_{2}$. The variable $x_{1}$ can take on the values of 1, 2, or 3, while $x_{2}$ can adopt the values of 4, 5, or 6.

| $x_{1}$ | $x_{2}$ |
|---------|---------|
| 1       | 4       |
| 2       | 5       |
| 3       | 6       |

This means that there are various combinations that these variables can form, thereby creating a comprehensive set or "exhaustive pool" of possibilities.

|    | 4     | 5     | 6     |
|----|-------|-------|-------|
| 1  | (1,4) | (1,5) | (1,6) |
| 2  | (2,4) | (2,5) | (2,6) |
| 3  | (3,4) | (3,5) | (3,6) |


### Example Code

```python
from autora.experimentalist.grid import grid_pool
from autora.variable import Variable, VariableCollection

iv_1 = Variable(allowed_values=[1, 2, 3])
iv_2 = Variable(allowed_values=[4, 5, 6])
variables = VariableCollection(independent_variables=[iv_1, iv_2])

pool = grid_pool(variables)
```
