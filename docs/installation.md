# Installation Guide

To install and use AutoRA you need:

- `Python` (version ">=3.8,<4") and
- the `autora` package, including required dependencies specified in the `pyproject.toml` file.

## Install `Python`

You can install `Python`:

- Using the instructions at [python.org](https://www.python.org), or
- Using a package manager, e.g.
  [homebrew](https://docs.brew.sh/Homebrew-and-Python), 
  [pyenv](https://github.com/pyenv/pyenv),
  [asdf](https://github.com/asdf-community/asdf-python), 
  [rtx](https://github.com/jdxcode/rtx/blob/main/docs/python.md),
  [winget](https://winstall.app/apps/Python.Python.3.8).

If successful, you should be able to run python in your terminal emulator like this:
```shell
python
```

...and see some output like this:
```
Python 3.11.3 (main, Apr  7 2023, 20:13:31) [Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```
## Install `autora`

To install the PyPI `autora` package, run the following command:

```shell
pip install "autora"
```

!!! success
    We recommended using a `Python` environment manager like `virtualenv`.

Check your installation by running:
```shell
python -c "from autora.variable import VariableCollection"
```

In using AutoRA, it is helpful to be aware of its structure, which is described next.

## Project Structure

AutoRA is organized into one "parent" and many "child" packages.

![image](img/package_overview.png)

[`autora`](https://github.com/autoresearch/autora) is the parent package which end users are expected to install (as specified above). The parent depends on core packages, such as [`autora-core`](https://github.com/autoresearch/autora-core), [`autora-workflow`](https://github.com/autoresearch/autora-workflow), and [`autora-synthetic`](https://github.com/autoresearch/autora-synthetic). It also includes vetted modules (child packages) as optional dependencies which users can choose to install.

For a complete list of optional dependencies that have been vetted by the core AutoRA team, see the `[project.optional-dependencies]` section of the `pyproject.toml` file in the parent `autora` package.

## Install Optional Dependencies

To install any (combintation) of optional dependencies, users should run the relevant analogue of the following command, with the name in brackets matching the name as specified in the parent `pyproject.toml` file:

```shell
pip install -U "autora[desired-dependency]"
```

For example, to install one of the [Theorists](theorist/index.md), such as the Bayesian Machine Scientist (BMS), a user should run:

```shell
pip install -U autora[theorist-bms]
```

To check that installation was successful, a user can try importing one of the main classes of the corresponding child package. For BMS, such a check would be:
```shell
python -c from autora.theorist.bms import BMSRegressor; BMSRegressor() 
```

