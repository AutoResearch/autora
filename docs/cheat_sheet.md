# Cheat Sheet

This cheat sheet provides code snippets for key concepts and functionalities of AutoRA. You may use this as a resource for developing AutoRA workflows.

## Installation

[Installation Guide](installation.md)

### Installing the main package:

```shell
pip install "autora"
```

### Installing Optional Packages

e.g., for ``autora-theorist-bms`` package:
```shell
pip install -U "autora[theorist-bms]"
```

## AutoRA Variables

[Variables Guide](https://autoresearch.github.io/autora/core/docs/Variable/)

### Defining Variables

```python
from autora.variable import VariableCollection, Variable
import numpy as np

variables = VariableCollection(
    independent_variables=[
        Variable(name="intensity", allowed_values[1, 2, 3, 4, 5]),
        Variable(name="duration", allowed_values=np.linspace(1, 100, 100))
    ],
    dependent_variables=[Variable(name="accuracy", value_range=(0, 1))]
)
```

### Extracting Variable Names

```python
ivs = [iv.name for iv in variables.independent_variables]
dvs = [dv.name for dv in variables.dependent_variables]
```

## State

[State Guide](https://autoresearch.github.io/autora/core/docs/The%20State%20Mechanism/)

### Defining Standard State

```python
from autora.state import StandardState
state = StandardState(
    variables=variables,
)
```

### Defining Custom State

```python
from autora.state import StandardState
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MyCustomState(StandardState):
    additional_field:  int = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )

# initialize the state:
state = MyCustomState(variables=variables)
```

### Retrieving Data From State

#### Conditions

```python
conditions = state.conditions
```

or 

```python
ivs = [iv.name for iv in variables.independent_variables]
conditions = state.experiment_data[ivs]
``` 

#### Experiment Data

```python
experiment_data = state.experiment_data
```

#### Observations

```python
experiment_data = state.experiment_data

dvs = [dv.name for dv in variables.dependent_variables]
observations = experiment_data[dvs]
```

#### Models
```python
last_model = state.model[-1]
```

### Defining State Wrappers

#### Theorist Wrapper 
```python
from autora.state import on_state, Delta

@on_state()
def theorist_on_state(experiment_data, variables):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]
    return Delta(models=[my_theorist.fit(x, y)])
```

#### Experimentalist Wrapper
```python
from autora.state import on_state, Delta

@on_state()
def experimentalist_on_state(allowed_conditions, num_samples):
    return Delta(conditions=my_experimentalist(allowed_conditions, num_samples))
```

#### Experiment Runner Wrapper
```python
from autora.state import on_state, Delta

on_state()
def experiment_runner_on_state(conditions, added_noise):
    return Delta(experiment_data=my_experiment_runner(conditions, added_noise))
```

### Calling State Wrappers

```python
state = runner_on_state(state)
```

!!! warning
    When adding your own input arguments to the wrapper, be sure to call the wrapper with all arguments specified in the function signature, e.g., for 
    ```python 
    @on_state()
    def experimentalist_on_state(conditions, num_samples):
        return Delta(conditions=my_experimentalist(allowed_conditions, num_samples))
    ```
    ``conditions`` is a variable retrieved from the state and ``num_samples`` is a custom argument. Thus, you must call the wrapper with `experimentalist_on_state(state, num_samples=num_samples)`. Note that `experimentalist_on_state(state, num_samples)` will throw an error.

### Running a Basic Workflow

[Workflow Guide](tutorials/basic/Tutorial III Functional Workflow.ipynb)

#### Without State Wrappers

```python
conditions = initial_experimentalist(state.variables)

for cycle in range(num_cycles):
    observations = experiment_runner(conditions, added_noise=1.0)
    model = theorist(conditions, observations)
    conditions = experimentalist(model, conditions, observations, num_samples=10)
```

#### With State Wrappers

```python
state = initial_experimentalist_on_state(state)

for cycle in range(num_cycles):
    state = experiment_runner_on_state(state, num_samples=10)
    state = theorist_on_state(state)
    state = experimentalist_on_state(state, model=state.model[-1])
```

## AutoRA Components

[Components Guide](tutorials/basic/Tutorial I Components.ipynb)

