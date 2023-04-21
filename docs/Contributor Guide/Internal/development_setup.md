# Getting started

You should be familiar with the command line for your operating system. The topics required are covered in:
- **macOS**: Joe Kissell. [*Take Control of the Mac Command Line with Terminal, 3rd Edition*](https://bruknow.library.brown.edu/permalink/01BU_INST/528fgv/cdi_safari_books_v2_9781947282513). Take Control Books, 2022. Chapters *Read Me First* through *Bring the Command Line Into The Real World*.
- **Linux**: William E. Shotts. [*The Linux Command Line: a Complete Introduction. 2nd edition.*](https://bruknow.library.brown.edu/permalink/01BU_INST/9mvq88/alma991043239704906966). No Starch Press, 2019. Parts *I: Learning the Shell* and *II: Configuration and the Environment*.

To use the AutoRA package you need:
- `python` and packages as specified in the `pyproject.toml` file,
- `graphviz` for some visualizations.

To develop the AutoRA package, you also need:
- `git`, the source control tool,
- `pre-commit` which is used for handling git pre-commit hooks.

We recommend setting up your development environment using:
- `pyenv` which is used for installing different versions of `python`,
- `poetry`, which handles resolving dependencies between `python` modules and ensures that you are using the same package versions as other members of the development team.

You should also consider using an IDE. We recommend: 
- PyCharm (academic licenses for PyCharm professional edition are available for free). This is a `python`-specific integrated development environment which comes with extremely powerful tools for changing the structure of `python` code, running tests, etc. 
- Visual Studio Code (free). This is a powerful general text editor with plugins to support `python` development. 

The following sections describe how to install and configure the recommended setup for developing AutoRA.

*Note: For end-users, it may be more appropriate to use an environment manager like `Anaconda` or `Miniconda` instead of `poetry`, but this is not currently supported.*


## Development Setup on macOS

### Prerequisites

For macOS, we strongly recommend using `homebrew` to manage packages.

Visit [https://brew.sh](https://brew.sh) and run the installation instructions.

### Clone Repository

We recommend using the GitHub CLI to clone the repository. Install it: 

```shell
brew install gh
```

Clone the repository. Run:
```shell
gh repo clone AutoResearch/AutoRA
```

... and following the prompts to authenticate to GitHub. It should clone the repository to a new directory. This is referred to as the `<project directory>` in the rest of this readme.

### Install Dependencies

Open the repository directory in the terminal.

Install the dependencies, which are listed in the [`Brewfile`](./Brewfile) by running:

```shell
brew bundle
```

### Install `python`

We recommend using `pyenv` to manage `python` versions. 

#### Initialize pyenv
Run the initialization script as follows:

```shell
pyenv init
``` 
... and follow the instructions to add `pyenv` to the `$PATH` by editing the interactive shell configuration 
file, `.zshrc` or `.bashrc`. If it exists, this file is a hidden file ([dotfile](https://missing.csail.mit.edu/2019/dotfiles/)) in your home directory. You can create or edit this file using a 
text editor or with CLI commands. Add the lines of script from the `pyenv init` response to the `.zshrc` file if they are 
not already present. 

#### Restart shell session

After making these changes, restart your shell session by executing:

```shell
exec "$SHELL" 
```

#### Install `python`

Install a `python` version listed in the [`pyproject.toml`](./pyproject.toml) file. The entry looks like:  

```toml
python = "^3.8”
```

In this case, you could install version 3.8.13 as follows:

```shell
pyenv install 3.8.13
```

### Install Pre-Commit Hooks

If you wish to commit to the repository, you should install the pre-commit hooks with the following command: 
```shell
pre-commit install
```

For more information on pre-commit hooks, see [Pre-Commit-Hooks](#pre-commit-hooks)

### Configure your development environment

There are two suggested options for initializing an environment:
- _(Recommended)_ Using PyCharm,
- _(Advanced)_ Using `poetry` from the command line.

#### PyCharm configuration

Set up the Virtual environment – an isolated version of `python` and all the packages required to run AutoRA and develop it further – as follows:
- Open the `<project directory>` in PyCharm.
- Navigate to PyCharm > Preferences > Project: AutoRA > Python Interpreter
- Next to the drop-down list of available interpreters, click the "Add Interpreter" and choose "Add Local Interpreter" to initialize a new interpreter. 
- Select "Poetry environment" in the list on the left. Specify the following:  
  - Python executable: select the path to the installed `python` version you wish to use, e.g. 
    `~/.pyenv/versions/3.8.13/bin/python3`
  - Select "install packages from pyproject.toml"
  - Poetry executable: select the path to the poetry installation you have, e.g. 
    `/opt/homebrew/bin/poetry`
  - Click "OK" and wait while the environment builds.
  - Run the "Python tests in tests/" Run/Debug configuration in the PyCharm interface, and check that there are no errors.

Additional setup steps for PyCharm:

- You can (and should) completely hide the IDE-specific directory for Visual Studio Code in PyCharm by adding `.vscode` to the list of ignored folder names in Preferences > Editor > File Types > Ignored Files and Folders. This only needs to be done once.
    
#### Command Line `poetry` Setup

If you need more control over the `poetry` environment, then you can set up a new environment from the command line.

*Note: Setting up a `poetry` environment on the command line is the only option for VSCode users.*

From the `<project directory>`, run the following commands.

Activate the target version of `python` using `pyenv`:
```shell
pyenv shell 3.8.13
```

Set up a new poetry environment with that `python` version:
```shell
poetry env use $(pyenv which python) 
```

Update the installation utilities within the new environment:
```shell
poetry run python -m pip install --upgrade pip setuptools wheel
```

Use the `pyproject.toml` file to resolve and then install all the dependencies
```shell
poetry install
```

Once this step has been completed, skip to the section [Activating and using the environment](#activating-and-using-the-environment) to test it.

#### Visual Studio Code Configuration

After installing Visual Studio Code and the other prerequisites, carry out the following steps:

- Open the `<project directory>` in Visual Studio Code
- Install the Visual Studio Code plugin recommendations suggested with the project. These include:
  - `python`
  - `python-environment-manager`
- Run the [Command Line poetry Setup](#command-line-poetry-setup) specified above. This can be done in the built-in terminal if desired (Menu: Terminal > New Terminal).
- Select the `python` option in the vertical bar on the far left of the window (which appear after installing the plugins). Under the title "PYTHON: ENVIRONMENTS" should be a list of `python` environments. If these do not appear:
  - Refresh the window pane
  - Ensure the python-environment-manager is installed correctly.
  - Ensure the python-environment-manager is activated.

- Locate the correct `poetry` environment. Click the "thumbs up" symbol next to the poetry environment name to "set as active workspace interpreter".

- Check that the `poetry` environment is correctly set-up. 
  - Open a new terminal within Visual Studio Code (Menu: Terminal > New Terminal). 
  - It should execute something like `source /Users/me/Library/Caches/pypoetry/virtualenvs/autora-2PgcgopX-py3.8/bin/activate` before offering you a prompt.
  - If you execute `which python` it should return the path to your python executable in the `.../autora-2PgcgopX-py3.8/bin` directory.
  - Ensure that there are no errors when you run: 
    ```shell
    python -m unittest
    ```
    in the built-in terminal. 

### Activating and using the environment

#### Using `poetry` interactively

To run interactive commands, you can activate the poetry virtual environment. From the `<project directory>` directory, run:

```shell
poetry shell
```

This spawns a new shell where you have access to the poetry `python` and all the packages installed using `poetry install`. You should see the prompt change:

```
% poetry shell
Spawning shell within /Users/me/Library/Caches/pypoetry/virtualenvs/autora-2PgcgopX-py3.8
Restored session: Fri Jun 24 12:34:56 EDT 2022
(autora-2PgcgopX-py3.8) % 
```

If you execute `python` and then `import numpy`, you should be able to see that `numpy` has been imported from the `autora-2PgcgopX-py3.8` environment:

```
(autora-2PgcgopX-py3.8) % python
Python 3.8.13 (default, Jun 16 2022, 12:34:56) 
[Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> numpy
<module 'numpy' from '/Users/me/Library/Caches/pypoetry/virtualenvs/autora-2PgcgopX-py3.8/lib/python3.8/site-packages/numpy/__init__.py'>
```

To deactivate the `poetry` environment, `exit` the session. This should return you to your original prompt, as follows:
```
(autora-2PgcgopX-py3.8) % exit

Saving session...
...saving history...truncating history files...
...completed.
% 
```

To run a script, e.g. the `weber.py` script in the [`example/sklearn/darts`](./example/sklearn/darts) directory, execute: 

```shell
poetry run python example/sklearn/darts/weber.py
```

#### Using `poetry` non-interactively

You can run python programs using poetry without activating the poetry environment, by using `poetry run {command}`. For example, to run the tests, execute:

```shell
poetry run python -m unittest
```

It should return something like:

```
% poetry run python -m unittest
.
--------------------------------
Ran 1 test in 0.000s

OK
```

## Development Setup on Windows

Windows is not yet officially supported. You may be able to follow the same approach as for macOS to set up your development environment, with some modifications, e.g.:
- Using `chocolatey` in place of `homebrew`,
- Using the GitHub Desktop application in place of the GitHub CLI.

If you successfully set up AutoRA on Windows, please update this readme.
