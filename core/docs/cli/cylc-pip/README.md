# Usage with Cylc workflow manager

The command line interface can be used with workflow managers like cylc in virtualenv environments.

!!! note
    This page covers basic usage of cylc. For usage with Slurm, see [cylc-slurm-pip](../cylc-slurm-pip) 

## Prerequisites

This example requires:

- familiarity with and a working installation of `cylc` (e.g. by going through the
  [tutorial](https://cylc.github.io/cylc-doc/latest/html/tutorial/index.html))
- `virtualenv`
- `python3.8` (so you can run `virtualenv venv -p python3.8`)

A new environment will be created during the setup phase of the `cylc` workflow run.

## Setup

To initialize the workflow, we define a file in the`lib/python` directory 
[(a cylc convention)](https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/configuration.html#workflow-configuration-directories) with the code for the experiment: 
[`lib/python/components.py`](./lib/python/components.py), including all the required functions. 

```python
--8<-- "https://raw.githubusercontent.com/AutoResearch/autora-workflow/main/docs/cli/cylc-pip/lib/python/components.py"
```

These functions will be called in turn by the `autora.workflow` script.

The [`flow.cylc`](flow.cylc) file defines the workflow.

```ini
--8<-- "https://raw.githubusercontent.com/AutoResearch/autora-workflow/main/docs/cli/cylc-pip/flow.cylc"
```

Note that the first step – `setup_python` – initializes a new virtual environment for python, using the requirements 
file. In this example, we require the following requirements, but yours will likely be different:

```ini
--8<-- "https://raw.githubusercontent.com/AutoResearch/autora-workflow/main/docs/cli/cylc-pip/requirements.txt"
```


## Execution

We can call the `cylc` command line interface as follows, in a shell session:

First, we validate the `flow.cylc` file:
```shell
cylc validate .
```

We install the workflow:
```shell
cylc install .
```

We tell cylc to play the workflow:
```shell
cylc play "cylc-pip"
```

(As a shortcut for "validate, install and play", use `cylc vip .`)

We can view the workflow running in the graphical user interface (GUI):
```shell
cylc gui
```

... or the text user interface (TUI):
```shell
cylc tui "cylc-pip"
```

## Results

We can load and interrogate the results as follows:

```python

from autora.serializer import load_state

state = load_state("~/cylc-run/cylc-pip/runN/share/result")
print(state)
```
