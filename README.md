# Autonomous Empirical Research
Autonomous Empirical Research is an open source AI-based system for automating each aspect empirical research in the behavioral sciences, from the construction of a scientific hypothesis to conducting novel experiments.

# Contributors (Alphabetic Order)
Ben Andrew, Hannah Even, Ioana Marinescu, Sebastian Musslick, Sida Li

# Getting started

To use the AER package you need:
- `python` and packages as specified in the `pyproject.toml` file,
- `graphviz` for some visualizations.

To develop the AER package, you also need:
- `git`, the source control tool,
- `pre-commit` which is used for handling git pre-commit hooks.

We recommend setting up your development environment using:
- `pyenv` which is used for installing different versions of `python`,
- `poetry`, which handles resolving dependencies between `python` modules and ensures that you are using the same package versions as other members of the development team.

You should also consider using an IDE. We recommend: 
- PyCharm (academic licenses for PyCharm professional edition are available for free). This is a `python`-specific integrated development environment which comes with extremely powerful tools for changing the structure of `python` code, running tests, etc. 
- Visual Studio Code (free). This is a powerful general text editor with plugins to support `python` development. 

The following sections describe how to install and configure the recommended setup recommended for developing AER.

*Note: For end-users, it may be more appropriate to use an environment manager like `Anaconda` or `Miniconda` instead of `poetry`, but this is not currently supported.*


## Development Setup on macOS

### Prerequisites

For macOS we strongly recommend using `homebrew` to manage packages.

Visit [https://brew.sh](https://brew.sh) and run the installation instructions.

### Clone Repository

We recommend using the GitHub CLI to clone the repository. Install it: 

```shell
brew install gh
```

Clone the repository. Run:
```shell
gh repo clone AutoResearch/AER
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
... then follow the instructions and add some lines to your shell environment, modifying the following files:
- If you use `zsh`, you'll modify `~/.zshrc` and `~/.zprofile`, 
- If you use `bash`, you'll modify `~/.bash_profile`.

#### Restart shell session

After making these changes, restart your shell session by executing:

```shell
exec "$SHELL" 
```

#### Install `python`

Install a `python` version listed in the [`pyproject.toml`](./pyproject.toml) file. The entry looks like:  

```toml
python = '>=3.8.13,<3.11'
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

Set up the Virtual environment – an isolated version of `python` and all the packages required to run AER and develop it further – as follows:
- Open the `<project directory>` in PyCharm.
- Navigate to PyCharm > Preferences > Project: AER > Python Interpreter
- Next to the drop-down list of available interpreters, click the "gear" symbol and choose "Add" to initialize a new interpreter. 
- Select "Poetry environment" in the list on the left. Specify the following:  
  - Python executable: select the path to the installed `python` version you wish to use, e.g. 
    `~/.pyenv/versions/3.8.13/bin/python3`
  - Select "install packages from pyproject.toml"
  - Poetry executable: select the path to the poetry installation you have, e.g. 
    `/opt/homebrew/bin/poetry`
  - Click "OK" and wait while the environment builds.
  - Run the "Python tests for aer" Run/Debug configuration in the PyCharm interface, and check that there are no errors.

Additional setup steps for PyCharm:

- You can (and should) completely hide the IDE-specific directory for Visual Studio Code in PyCharm by adding `.vscode` to the list of ignored folder names in Preferences > Editor > File Types > Ignored Files and Folders. This only needs to be done once.
    
#### Command Line `poetry` Setup

If you need more control over the `poetry` environment than offered by PyCharm, then you can set up a new environment from the command line as follows. From the `<project directory>`, run the following commands:

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

##### Using `poetry` interactively

To run interactive commands, you can activate the poetry virtual environment. From the `<project directory>` directory, run:

```shell
poetry shell
```

This spawns a new shell where you have access to the poetry `python` and all the packages installed using `poetry install`. You should see the prompt change:

```
% poetry shell
Spawning shell within /Users/me/Library/Caches/pypoetry/virtualenvs/aer-2PgcgopX-py3.8
Restored session: Fri Jun 24 12:34:56 EDT 2022
(aer-2PgcgopX-py3.8) % 
```

If you execute `python` and then `import numpy`, you should be able to see that `numpy` has been imported from the `aer-2PgcgopX-py3.8` environment :

```
(aer-2PgcgopX-py3.8) % python
Python 3.8.13 (default, Jun 16 2022, 12:34:56) 
[Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> numpy
<module 'numpy' from '/Users/me/Library/Caches/pypoetry/virtualenvs/aer-2PgcgopX-py3.8/lib/python3.8/site-packages/numpy/__init__.py'>
```

To deactivate the `poetry` environment, `exit` the session. This should return you to your original prompt, as follows:
```
(aer-2PgcgopX-py3.8) % exit

Saving session...
...saving history...truncating history files...
...completed.
% 
```

##### Using `poetry` non-interactively

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

#### Visual Studio Code Configuration

After installing Visual Studio Code and the other prerequisites, carry out the following steps:

- Open the `<project directory>` in Visual Studio Code
- Install the Visual Studio Code plugin recommendations suggested with the project. These include:
  - `python`
  - `python-environment-manager`
- Run the [Command Line poetry Setup](#command-line-poetry-setup) specified above. This can be done in the built-in terminal if desired (Menu: Terminal > New Terminal).
- Select the `python` option in the vertical bar on the far left of the window (which appear after installing the plugins). Under the title "PYTHON: ENVIRONMENTS" should be a list of `python` environments. If these do not appear:
  - Ensure the python-environment-manager is installed correctly.
  - Ensure the python-environment-manager is activated.

- Locate the correct `poetry` environment. Click the "thumbs up" symbol next to the poetry environment name to "set as active workspace interpreter".

- Check that the `poetry` environment is correctly set-up. 
  - Open a new terminal within Visual Studio Code (Menu: Terminal > New Terminal). 
  - It should execute something like `source /Users/me/Library/Caches/pypoetry/virtualenvs/aer-2PgcgopX-py3.8/bin/activate` before offering you a prompt.
  - If you execute `which python` it should return the path to your python executable in the `.../aer-2PgcgopX-py3.8/bin` directory.
  - Ensure that there are no errors when you run: 
    ```shell
    python -m unittest
    ```
    in the built-in terminal. 

## Development Setup on Windows

Windows is not yet officially supported. You may be able to follow the same approach as for macOS to set up your development environment, with some modifications, e.g.:
- Using `chocolatey` in place of `homebrew`,
- Using the GitHub Desktop application in place of the GitHub CLI.

If you successfully set up AER on Windows, please update this readme.

Please update this readme 

## Development Practices

### Running the tests

You should run the tests before you commit code to the repository, to ensure that you've not broken anything. 

The unit tests can be run as follows (starting in the root directory of the repository):

```shell
poetry run python -m unittest
```

You can also use the run configuration "Python tests for aer" in PyCharm.

### Pre-Commit Hooks

We use [`pre-commit`](https://pre-commit.com) to manage pre-commit hooks. 

Pre-commit hooks are programs which run before each git commit, and can read and potentially modify the files which are to be committed. 

We use pre-commit hooks to:
- enforce coding guidelines, including the `python` style-guide [PEP8](https://peps.python.org/pep-0008/) (`black` and `flake8`), 
- to check the order of `import` statements (`isort`),
- to check the types of `python` objects (`mypy`).

The hooks and their settings are specified in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

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
   This might mean unstaging changes and then adding them again.**
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
