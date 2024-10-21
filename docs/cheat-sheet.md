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

## AutoRA Components

[Components Guide](tutorials/basic/Tutorial I Components.ipynb)

### Theorists

[Theorist Overview](theorist/index.md)

#### Fit & Predict
```python
from autora.theorist.bms import BMSRegressor

# declare theorist
theorist = BMSRegressor(epochs=100)

# fit theorist to data
model = theorist.fit(conditions, observations)

# predict new observations
observations = theorist.predict(conditions)
```

#### Write Custom Theorist

[Custom Theorist Guide](contribute/modules/theorist.md)

```python
from sklearn.base import BaseEstimator

class LogisticRegressor(BaseEstimator):
    def __init__(self, *args, **kwargs):
        self.model = MyTheoristMethod(*args, **kwargs)  

    def fit(self, conditions, observations):
        self.model.fit(conditions, observations)
        return self

    def predict(self, conditions):
        return self.model.predict(observations) 
```

`conditions` should be a pandas DataFrame with columns corresponding to the independent variables.

`observations` should be a pandas DataFrame with columns corresponding to the dependent variables.

### Experimentalists

[Experimentalist Overview](experimentalist/index.md)

#### Generate Conditions
```python
from autora.experimentalist.random import random_pool

conditions = random_pool(variables, num_samples=10)
```

#### Write Custom Experimentalist

[Custom Experimentalist Guide](contribute/modules/experimentalist.md)

```python
def my_experimentalist(allowed_conditions, num_samples):
    # ...
    return selected_conditions
```

`conditions` should be a pandas DataFrame with columns corresponding to the independent variables.

### Experiment Runners

#### Run Experiment
```python
experiment_runner = weber_fechner_law()
experiment_data = experiment_runner.run(conditions)
```

`conditions` should be a pandas DataFrame with columns corresponding to the independent variables.

``experiment_data`` should be a pandas DataFrame with columns corresponding to the dependent variables.

#### Using Synthetic Experiment Runners

Equation Runner Example:
```python
from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from sympy import symbols
import numpy as np

x, y = symbols("x y")
expr = x ** 2 - y ** 2

experiment = equation_experiment(expr)

test_input = np.array([[1, 1], [2 ,2], [2 ,3]])

experiment.experiment_runner(test_input)
```

Weber-Fechner Example:

```python
# synthetic experiment from autora inventory
from autora.experiment_runner.synthetic.psychophysics.weber_fechner_law import weber_fechner_law

synthetic_runner = weber_fechner_law(constant=3)

variables = synthetic_runner.variables
conditions = synthetic_runner.domain()
experiment_data = synthetic_runner.run(conditions, added_noise=0.01)
```

#### Using Behavioral Experiment Runners

[Example Study Guide](examples/closed-loop-basic/index.md)

##### Initializing and Running Firebase Runner

```python
firebase_credentials = {
  "type": "service_account",
  "project_id": "closed-loop-study",
  "private_key_id": "YOURKEYID",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOURCREDENTIALS\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-y7hnh@closed-loop-study.iam.gserviceaccount.com",
  "client_id": "YOURLIENTID",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-y7hnh%40closed-loop-study.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=5,
    sleep_time=3)

data_raw = experiment_runner(conditions_to_send)
```

##### Initializing and Running Prolific Runner

```python
sleep_time = 30
study_name = 'my autora experiment'
study_description= 'Psychophysics Study'
study_url = 'https://closed-loop-study.web.app/'
study_completion_time = 5
prolific_token = 'my prolific token'
completion_code = 'my completion code'

experiment_runner = firebase_prolific_runner(
            firebase_credentials=firebase_credentials,
            sleep_time=sleep_time,
            study_name=study_name,
            study_description=study_description,
            study_url=study_url,
            study_completion_time=study_completion_time,
            prolific_token=prolific_token,
            completion_code=completion_code,
        )
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
    ``conditions`` is a variable retrieved from the state and ``num_samples`` is a custom argument. Thus, you must call the wrapper with 
    ```python 
    experimentalist_on_state(state, num_samples=num_samples)
    ```
    Note that `experimentalist_on_state(state, num_samples)` will throw an error.

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
