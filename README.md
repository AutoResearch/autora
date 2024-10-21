# Automated Research Assistant

![PyPI](https://img.shields.io/pypi/v/autora)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/autoresearch/autora/test-pytest.yml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/autora)
![Link to docs](https://img.shields.io/badge/Docs-autoresearch.github.io-purple)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub Discussions](https://img.shields.io/github/discussions/autoresearch/autora)
[![DOI](https://zenodo.org/badge/261852986.svg)](https://zenodo.org/doi/10.5281/zenodo.10277414)
[![status](https://joss.theoj.org/papers/be6d470033fbe5bd705a49858eb4e21e/status.svg)](https://joss.theoj.org/papers/be6d470033fbe5bd705a49858eb4e21e)

<a href="https://ccbs.carney.brown.edu/brainstorm"><img src="docs/img/brainstorm.png" alt="BRAINSTORM Program" height="60"></img></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://schmidtsciencefellows.org/"><img src="docs/img/ssf.png" alt="Schmidt Science Fellows" height="60"></img></a>

<b>[AutoRA](https://pypi.org/project/autora/)</b> (<b>Auto</b>mated <b>R</b>esearch <b>A</b>ssistant) is an open-source framework for 
automating multiple stages of the empirical research process, including model discovery, experimental design, data collection, and documentation for open science. 

AutoRA was initially intended for accelerating research in the behavioral and brain sciences. However, AutoRA is designed as a general framework that enables automation of the research processes in other empirical sciences, such as material science or physics.

![Autonomous Empirical Research Paradigm](https://github.com/AutoResearch/autora/raw/main/docs/img/overview.png)

## Installation


We recommend using a `Python` environment manager like `virtualenv`. You may refer to the Development Guide on how to [set up a virtual environment](https://autoresearch.github.io/autora/contribute/setup/#create-a-virtual-environment).  

Before installing the PyPI ``autora`` package, you may [activate your environment](https://autoresearch.github.io/autora/contribute/setup/#activating-and-using-the-environment). To install the PyPI `autora` package, run the following command:

```shell
pip install "autora"
```

## Documentation

Check out tutorials and documentation at 
[https://autoresearch.github.io/autora](https://autoresearch.github.io/autora). If you run into any issues or questions regarding the use of AutoRA, please reach out to us at the [AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/using-autora).

## Example

The following basic example demonstrates how to use AutoRA to automate the process of model discovery, experimental design, and data collection. 

The discovery problem is defined by a single independent variable $x \in [0, 2 \pi]$ and dependent variable $y$.
The experiment amounts to a simple sine wave, $y = \sin(x)$, which is the model we are trying to discover.

Th discovery cycle iterates between the experimentalist, experiment runner, and theorist. Here, we us a "random" experimentalist, which samples novel experimental conditions for $x$ every cycle. 
The experiment runner then collects data for the corresponding $y$ values. Finally, the theorist uses a [Bayesian Machine Scientist](https://autoresearch.github.io/autora/user-guide/theorists/bms/) (BMS; Guimerà et al., in Science Advances) to identify a scientific model that explains the data. 

The workflow relies on the ``StandardState`` object, which stores the current state of the discovery process, such as ``conditions``, ``experiment_data``, or ``models``. The state is passed between the experimentalist, experiment runner, and theorist.


```python
####################################################################################
## Import statements
####################################################################################

import pandas as pd 
import numpy as np
import sympy as sp

from autora.variable import Variable, ValueType, VariableCollection

from autora.experimentalist.random import random_pool
from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from autora.theorist.bms import BMSRegressor

from autora.state import StandardState, on_state, estimator_on_state

####################################################################################
## Define initial data
####################################################################################

#### Define variable data ####
iv = Variable(name="x", value_range=(0, 2 * np.pi), allowed_values=np.linspace(0, 2 * np.pi, 30))
dv = Variable(name="y", type=ValueType.REAL)
variables = VariableCollection(independent_variables=[iv],dependent_variables=[dv])

#### Define seed condition data ####
conditions = random_pool(variables, num_samples=10, random_state=0)

####################################################################################
## Define experimentalist
####################################################################################

experimentalist = on_state(random_pool, output=["conditions"])

####################################################################################
## Define experiment runner
####################################################################################

sin_experiment = equation_experiment(sp.simplify('sin(x)'), variables.independent_variables, variables.dependent_variables[0])
sin_runner = sin_experiment.experiment_runner

experiment_runner = on_state(sin_runner, output=["experiment_data"])

####################################################################################
## Define theorist
####################################################################################

theorist = estimator_on_state(BMSRegressor(epochs=100))

####################################################################################
## Define state
####################################################################################

s = StandardState(
    variables = variables,
    conditions = conditions,
    experiment_data = pd.DataFrame(columns=["x","y"])
)

####################################################################################
## Cycle through the state
####################################################################################

print('Pre-Defined State:')
print(f"Number of datapoints collected: {len(s['experiment_data'])}")
print(f"Derived models: {s['models']}")
print('\n')

for i in range(5):
    s = experimentalist(s, num_samples=10, random_state=42)
    s = experiment_runner(s, added_noise=1.0, random_state=42)
    s = theorist(s)
    print(f"\nCycle {i+1} Results:")
    print(f"Number of datapoints collected: {len(s['experiment_data'])}")
    print(f"Derived models: {s['models']}")
    print('\n')
```

If you are curious about how to apply AutoRA to real-world discovery problems, you can find use case examples of AutoRA in the [Use Case Tutorials](https://autoresearch.github.io/autora/examples/) section of the documentation.

## Contributions

We welcome contributions to the AutoRA project. Please refer to the [contributor guide](https://autoresearch.github.io/autora/contribute/) for more information. Also, feel free to ask any questions or provide any feedback regarding core contributions on the [AutoRA forum](https://github.com/orgs/AutoResearch/discussions/). 

## About

This project is in active development by the [Autonomous Empirical Research Group](https://musslick.github.io/AER_website/Research.html).

The development of this package was supported by [Schmidt Science Fellows](https://schmidtsciencefellows.org/), in partnership with the Rhodes Trust, as well as the [Carney BRAINSTORM program](https://ccbs.carney.brown.edu/brainstorm) at Brown University. The development of auxiliary packages for AutoRA, such as `autodoc`, is supported by [Schmidt Sciences, LLC. and the Virtual Institute for Scientific Software (VISS)](https://www.schmidtsciences.org/viss/). The AutoRA package was developed using computational resources and services at the [Center for Computation and Visualization at Brown University](https://ccv.brown.edu).


## Read More

- [Package Documentation](https://autoresearch.github.io/autora/)
- [AutoRA Pip Package](https://pypi.org/project/autora/)
- [Autonomous Empirical Research Group](http://www.empiricalresearch.ai)

