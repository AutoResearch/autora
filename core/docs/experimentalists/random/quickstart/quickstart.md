# Quickstart Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)


*Random Pooler* and *Sampler* are part of the `autora-core` package and do not need to be installed separately

You can import and invoke the pool like this:

```python
from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool

pool(
    VariableCollection(independent_variables=[Variable(name="x", allowed_values=range(10))]),
    random_state=1
)
```

You can import the sampler like this:

```python
from autora.experimentalist.random import sample

sample([1, 1, 2, 2, 3, 3], num_samples=2)
```

