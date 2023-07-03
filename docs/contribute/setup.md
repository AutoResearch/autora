# Setup Guide

It's possible to set up your python environment in many different ways. 

To use the AutoRA package you need:

- `python` and 
- packages as specified in the `pyproject.toml` file.

To develop the AutoRA package, you also need:

- `git`, the source control tool,
- `pre-commit` which is used for handling git pre-commit hooks.

You should also consider using an IDE. We recommend: 

- PyCharm. This is a `python`-specific integrated development environment which comes with useful tools 
  for changing the structure of `python` code, running tests, etc. 
- Visual Studio Code. This is a powerful general text editor with plugins to support `python` development.

The following sections describe how to install and configure the recommended setup for developing AutoRA.

!!! tip 
    It is helpful to be familiar with the command line for your operating system. The topics required are covered in:

    - **macOS**: Joe Kissell. [*Take Control of the Mac Command Line with Terminal, 3rd Edition*](https://bruknow.library.brown.edu/permalink/01BU_INST/528fgv/cdi_safari_books_v2_9781947282513). Take Control Books, 2022. Chapters *Read Me First* through *Bring the Command Line Into The Real World*.
    - **Linux**: William E. Shotts. [*The Linux Command Line: a Complete Introduction. 2nd edition.*](https://bruknow.library.brown.edu/permalink/01BU_INST/9mvq88/alma991043239704906966). No Starch Press, 2019. Parts *I: Learning the Shell* and *II: Configuration and the Environment*.

## Development Setup

### Clone The Repository

The easiest way to clone the repo is to go to [the repository page on GitHub](https://github.com/AutoResearch/autora)
and click the "<> Code" button and follow the prompts. 

!!! hint
    We recommend using:
    
    - the [GitHub Desktop Application](https://desktop.github.com) on macOS or Windows, or 
    - the [GitHub command line utility](https://cli.github.com) on Linux.

### Install `Python`

!!! success
    All contributions to the AutoRA core packages should work under **python 3.8**, so we recommend using that version 
    for development.
    
You can install python:

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

#### Create A Virtual Environment

!!! success
    We recommend setting up your development environment using a manager like `venv`, which creates isolated python 
    environments. Other environment managers, like 
    [virtualenv](https://virtualenv.pypa.io/en/latest/),
    [pipenv](https://pipenv.pypa.io/en/latest/),
    [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), 
    [hatch](https://hatch.pypa.io/latest/), 
    [poetry](https://python-poetry.org), 
    are available and will likely work, but will have different syntax to the syntax shown here. 

    Our packages are set up using `virtualenv` with `pip`  

In the `<project directory>`, run the following command to create a new virtual environment in the `.venv` directory

```shell
python3 -m "venv" ".venv" 
```

!!! hint
    If you have multiple Python versions installed on your system, it may be necessary to specify the Python version when creating a virtual environment. For example, run the following command to specify Python 3.8 for the virtual environment. 
    ```shell
    python3.8 -m "venv" ".venv" 
    ```

Activate it by running
```shell
source ".venv/bin/activate"
```

#### Install Dependencies

Upgrade pip:
```shell
pip install --upgrade pip
```

Install the current project development dependencies:
```shell
pip install --upgrade --editable ".[dev]"
```

Your IDE may have special support for python environments. For IDE-specific setup, see:

- [PyCharm Documentation](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)
- [VSCode Documentation](https://code.visualstudio.com/docs/python/environments)


### Activating And Using The Environment

To run interactive commands, you can activate the virtualenv environment. From the `<project directory>` 
directory, run:

```shell
source ".venv/bin/activate"
```

This spawns a new shell where you have access to the `python` and all the packages installed using `pip install`. You 
should see the prompt change:

```
% source .venv/bin/activate
(.venv) % 
```


If you execute `python` and then `import numpy`, you should be able to see that `numpy` has been imported from the 
`.venv` environment:

```
(.venv) % python
Python 3.8.16 (default, Dec 15 2022, 14:31:45) 
[Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> numpy
<module 'numpy' from '/Users/me/Developer/autora/.venv/lib/python3.8/site-packages/numpy/__init__.py'>
>>> exit()
(.venv) %
```

You should be able to check that the current project works by running the tests:
```shell
pytest
```

It should return something like:

```
% pytest
.
--------------------------------
Ran 1 test in 0.000s

OK
```


!!! hint
    To deactivate the `virtualenv` environment, `deactivate` it. This should return you to your original prompt,
    as follows:
    ```
    (venv) % deactivate
    % 
    ```


### Running Code Non-Interactively

You can run python programs without activating the environment, by using `/path/to/python run {command}`. For example,
to run unittests tests, execute:

```shell
.venv/bin/python -m pytest
```

It should return something like:

```
% .venv/bin/python -m pytest
.
--------------------------------
Ran 1 test in 0.000s

OK
```

### Pre-Commit Hooks

If you wish to commit to the repository, you should install and activate `pre-commit` as follows. 
```shell
pip install pre-commit
pre-commit install
```

You can run the pre-commit hooks manually by calling:
```shell
pre-commit run --all-files
```

For more information on pre-commit hooks, see [Pre-Commit-Hooks](./pre-commit-hooks.md)

