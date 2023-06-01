# Setup Guide

It's possible to set up your python environment in many different ways. 

To use the AutoRA package you need:
- `python` and packages as specified in the `pyproject.toml` file,

!!! success
    We recommend setting up your development environment using `virtualenv`, which creates isolated python environments. 

To develop the AutoRA package, you also need:
- `git`, the source control tool,
- `pre-commit` which is used for handling git pre-commit hooks.

You should also consider using an IDE. We recommend: 
- PyCharm. This is a `python`-specific integrated development environment which comes with extremely powerful tools 
  for changing the structure of `python` code, running tests, etc. 
- Visual Studio Code. This is a powerful general text editor with plugins to support `python` development. 

The following sections describe how to install and configure the recommended setup for developing AutoRA.

- You should be familiar with the command line for your operating system. The topics required are covered in:
- **macOS**: Joe Kissell. [*Take Control of the Mac Command Line with Terminal, 3rd Edition*](https://bruknow.library.brown.edu/permalink/01BU_INST/528fgv/cdi_safari_books_v2_9781947282513). Take Control Books, 2022. Chapters *Read Me First* through *Bring the Command Line Into The Real World*.
- **Linux**: William E. Shotts. [*The Linux Command Line: a Complete Introduction. 2nd edition.*](https://bruknow.library.brown.edu/permalink/01BU_INST/9mvq88/alma991043239704906966). No Starch Press, 2019. Parts *I: Learning the Shell* and *II: Configuration and the Environment*.

## Development Setup on macOS

### Prerequisites

For macOS, we strongly recommend using `homebrew` to manage packages.

Visit [https://brew.sh](https://brew.sh) and run the installation instructions.

### Clone Repository

We recommend using the GitHub Desktop Application to clone the repository. Install it: 

```shell
brew install --cask github
```

Clone the repository by: 
- opening the GitHub Desktop application, then 
- opening File > Clone Repository and 
- selecting the repository you would like to clone. 

It may prompt you to authenticate, and should clone the repository to a new directory. This directory is referred 
to as the `<project directory>` in the rest of this readme.


### Install `python`

We recommend using `pyenv` to manage `python` versions. 

#### Initialize `pyenv`

Install `pyenv` by running:
```shell
brew install pyenv
```

Run its initialization script as follows:

```shell
pyenv init
``` 
... and follow the instructions to add `pyenv` to the `$PATH` by editing the interactive shell configuration 
file, `.zshrc` or `.bashrc`. 

!!! hint 
    If it exists, this file is a hidden file ([dotfile](https://missing.csail.mit.edu/2019/dotfiles/)) in your home 
    directory. You can create or edit this file using a text editor or with CLI commands. Add the lines of script from
    the `pyenv init` response to the `.zshrc` file if they are not already present. 

#### Restart shell session

After making these changes, restart your shell session by executing:

```shell
exec "$SHELL" 
```

#### Install `python`

Install `python` (version 3.8 or newer).

You could install version 3.8 as follows:

```shell
pyenv install 3.8
```

### Install Pre-Commit Hooks

If you wish to commit to the repository, you should install `pre-commit`: 
```shell
brew install pre-commit
```

Activate the pre-commit hooks as follows: 
```shell
pre-commit install
```

For more information on pre-commit hooks, see [Pre-Commit-Hooks](#pre-commit-hooks)

### Configure your development environment

There are two suggested options for initializing an environment:
- _(Recommended)_ Using PyCharm,
- _(Advanced)_ Using `pip` from the command line.

#### PyCharm configuration

Set up the Virtual environment – an isolated version of `python` and all the packages required to run AutoRA and develop it further – as follows:
- Open the `<project directory>` in PyCharm.
- Navigate to PyCharm > Preferences > Project: AutoRA > Python Interpreter
- Next to the drop-down list of available interpreters, click the "Add Interpreter" and choose "Add Local Interpreter" to initialize a new interpreter. 
- Select "virtualenv environment" in the list on the left. Specify the following:  
  - Environment: new
  - Location: `<project directory>/venv`
  - Python executable: the path to the installed `python` version you wish to use, e.g. 
    `~/.pyenv/versions/3.8.16/bin/python3`
  - Click "OK" and wait while the environment builds.

- Open a terminal in PyCharm.
  - Check that the virtualenvironment is active – run 
    ```shell
    which python
    ``` 
    which should return the path to the python in 
    your virtualenv, e.g. `<project directory>/venv/bin/python`. If it is not, activate the environment by running 
    ```shell
    source venv/bin/activate
    ```
  - Use `pip install` to install the current project (`"."`) in editable mode (`--editable`) 
    with dev-dependencies (`[dev]`),
    ensuring that everything is updated (`--upgrade`):
    ```shell
    pip install --upgrade --editable ".[dev]"
    ```
  - Run the tests using:
    ```shell
    pytest --doctest-modules
    ```
 
#### Command Line Setup

If you need more control over the `virtualenv` environment, then you can set up a new environment from the command line.

From the `<project directory>`, run the following commands.

Set up a new virtualenv environment with an installed version of `python` version in the `venv` directory:
```shell
virtualenv --python="/path/to/your/python/installation" venv
```

Activate the environment
```shell
source venv/bin/activate
```

Update the installation utilities within the new environment:
```shell
pip install --upgrade pip setuptools wheel
```

Install and upgrade all the dependencies:
```shell
pip install --upgrade --editable ".[dev]"
```

Once this step has been completed, skip to the section [Activating and using the environment](#activating-and-using-the-environment) to test it.

#### Visual Studio Code Configuration

After installing Visual Studio Code and the other prerequisites, carry out the following steps:

- Open the `<project directory>` in Visual Studio Code
- Install the Visual Studio Code plugin recommendations suggested with the project. These include:
  - `python`
  - `python-environment-manager`
- Run the [Command Line Setup](#command-line-setup) specified above. This can be done in the built-in terminal if desired (Menu: Terminal > New Terminal).
- Select the `python` option in the vertical bar on the far left of the window (which appear after installing the plugins). Under the title "PYTHON: ENVIRONMENTS" should be a list of `python` environments. If these do not appear:
  - Refresh the window pane
  - Ensure the python-environment-manager is installed correctly.
  - Ensure the python-environment-manager is activated.

- Locate the correct `virtualenv` environment. Click the "thumbs up" symbol next to the environment name to 
  "set as active workspace interpreter".

- Check that the `virtualenv` environment is correctly set-up. 
  - Open a new terminal within Visual Studio Code (Menu: Terminal > New Terminal). 
  - It should execute something like `source <project directory>/venv/bin/activate` before offering you a prompt.
  - If you execute `which python` it should return the path to your python executable in the `.../venv/bin` directory.
  - Ensure that there are no errors when you run: 
    ```shell
    pytest --doctest-modules
    ```
    in the built-in terminal. 

### Activating and using the environment

#### Using `virtualenv` interactively

To run interactive commands, you can activate the virtualenv environment. From the `<project directory>` 
directory, run:

```shell
source venv/bin/activate
```

This spawns a new shell where you have access to the `python` and all the packages installed using `pip install`. You 
should see the prompt change:

```
% source venv/bin/activate
(venv) % 
```

If you execute `python` and then `import numpy`, you should be able to see that `numpy` has been imported from the 
`venv` environment:

```
(venv) % python
Python 3.8.16 (default, Dec 15 2022, 14:31:45) 
[Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> numpy
<module 'numpy' from '/Users/me/Developer/autora/venv/lib/python3.8/site-packages/numpy/__init__.py'>
>>> exit()
(venv) %
```

!!! hint
    To deactivate the `virtualenv` environment, `deactivate` it. This should return you to your original prompt,
    as follows:
    ```
    (venv) % deactivate
    % 
    ```


#### Running code non-interactively

You can run python programs without activating the environment, by using `/path/to/python run {command}`. For example,
to run unittests tests, execute:

```shell
venv/bin/python -m pytest
```

It should return something like:

```
% venv/bin/python -m pytest
.
--------------------------------
Ran 1 test in 0.000s

OK
```

## Development Setup on Windows

Windows is not yet officially supported. You may be able to follow the same approach as for macOS to set up your 
development environment, with some modifications, e.g. using `winget` in place of `homebrew`,

If you successfully set up AutoRA on Windows, please update this readme.

## Development Practices

### Pre-Commit Hooks

We use [`pre-commit`](https://pre-commit.com) to manage pre-commit hooks. 

Pre-commit hooks are programs which run before each git commit, and can read and potentially modify the files which are to be committed. 

We use pre-commit hooks to:
- enforce coding guidelines, including the `python` style-guide [PEP8](https://peps.python.org/pep-0008/) (`black` and `flake8`), 
- to check the order of `import` statements (`isort`),
- to check the types of `python` objects (`mypy`).

The hooks and their settings are specified in the `.pre-commit-config.yaml` in each repository.

See the section [Install Pre-commit Hooks](#install-pre-commit-hooks) for installation instructions.

#### Handling Pre-Commit Hook Errors

If your `git commit` fails because of the pre-commit hook, then you should:

1. Run the pre-commit hooks on the files which you have staged, by running the following command in your terminal: 
    ```zsh
    $ pre-commit run
    ```

2. Inspect the output. It might look like this:
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Failed
   - hook id: mypy
   - exit code: 1
   
   example.py:33: error: Need type annotation for "data" (hint: "data: Dict[<type>, <type>] = ...")
   Found 1 errors in 1 files (checked 10 source files)
   ```
3. Fix any errors which are reported.
   **Important: Once you've changed the code, re-stage the files it to Git. 
   This might mean un-staging changes and then adding them again.**
4. If you have trouble:
   - Do a web-search to see if someone else had a similar error in the past.
   - Check that the tests you've written work correctly.
   - Check that there aren't any other obvious errors with the code.
   - If you've done all of that, and you still can't fix the problem, get help from someone else on the team.
5. Repeat 1-4 until all hooks return "passed", e.g.
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Passed
   ```

It's easiest to solve these kinds of problems if you make small commits, often.  
