# Autonomous Empirical Research
Autonomous Empirical Research is an open source AI-based system for automating each aspect empirical research in the behavioral sciences, from the construction of a scientific hypothesis to conducting novel experiments.

# Contributors (Alphabetic Order)
Ben Andrew, Hannah Even, Ioana Marinescu, Sebastian Musslick, Sida Li

# Getting started (development)

The following sections describe how to install and configure the supported setup recommended for developing AER, using:
- `python` and dependencies as described in the `pyproject.toml` file,
- `pyenv` (optional) which is used for installing different versions of `python`,
- `poetry` (mandatory) for managing the `python` environment itself,
- `pre-commit` (mandatory when making changes to the repository) which is used for handling git pre-commit hooks,
- `graphviz` (mandatory) which is used for some visualizations.

You should also consider using an IDE. We recommend: 
- PyCharm (academic licenses for PyCharm pro are available for free). This is a `python`-specific integrated development environment which comes with extremely powerful tools for changing the structure of `python` code, running tests, etc. 
- Visual Studio Code (free). This is a powerful general text editor with a large number of plugins available for different tasks. 

- Depending on your computer, the steps to get AER working will vary. At present, macOS is supported. 



## Basic Setup on MacOS 

### Install dependencies

#### Install `homebrew` (optional)

For macOS we strongly recommend using `homebrew` as the package manager.

Visit [https://brew.sh](https://brew.sh) and run the installation instructions.

#### Install external tools using `homebrew`

This is a good time to check that the other external dependencies are fulfilled. For developing AER you need:
- `pyenv` which is required for the `python` setup,
- `pre-commit` which is used for handling git pre-commit hooks,
- `graphviz` which is used for some visualizations.

You can install them as follows:

```shell
brew bundle
```

This uses the [`Brewfile`](./Brewfile) to install all the packages required.

#### Initialize pyenv

`pyenv` allows installing different versions of `python`. Run the initialization script as follows:

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

## Set up the `python` environment

### Install `python` version 

Install a `python` version listed in the [`pyproject.toml`](./pyproject.toml) file. The entry looks like:  

```toml
python = '>=3.8.13,<3.11'
```

In this case, you could install version 3.8.13 as follows:

```shell
pyenv install 3.8.13
```

### Set up the `poetry` environment

You can use this `python` version with the `poetry` package manager to create an isolated environment where you can run the AER code. `poetry` handles resolving dependencies between `python` modules and ensures that you are using the same package versions as other members of the development team (which is a good thing).

There are two suggested options for initializing an environment:
- On the command line using `poetry` directly,
- Using your IDE to initialize the `poetry` environment. 

*Note: For end-users, it may be more appropriate to use an environment manager like `Anaconda` or `Miniconda` instead of `poetry`, but this is not currently supported.*

#### Command Line `poetry` Setup

From the [`AER`](./.) directory, run:

```zsh
# Optional: configure poetry to save the environment in the AER/.venv/ directory. 
# We'll assume you've done this in the rest of the description.
poetry config virtualenvs.in-project true

# Set up a new environment with the version of python you installed earlier
pyenv shell 3.8.13  # Activate the target version of python
poetry env use $(pyenv which python)  # Set up a new poetry environment with that python version

# Update the installation utilities within the new environment
poetry run python -m pip install --upgrade pip setuptools wheel

# Use the pyproject.toml file to resolve and then install all of the dependencies
poetry install
```

##### Using `poetry` interactively

To run interactive commands, you can activate the poetry virtual environment. From the [`AER`](./.) directory, run:

```shell
% poetry shell
```

This spawns a new shell where you have access to the poetry `python` and all of the packages installed using `poetry install`. You should see the prompt change:

```shell
% poetry shell
Spawning shell within /Users/me/Developer/AER/.venv
Restored session: Fri Jun 24 12:34:56 EDT 2022
(.venv) % 
```

If you execute `python` and then `import numpy`, you should be able to see that `numpy` has been imported from the `.venv` directory :

```shell
(.venv) % python
Python 3.8.13 (default, Jun 16 2022, 12:34:56) 
[Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> numpy
<module 'numpy' from '.../AER/.venv/lib/python3.8/site-packages/numpy/__init__.py'>
```

To deactivate the `poetry` environment, `exit` the session. This should return you to your original prompt, as follows:
```shell
(.venv) % exit

Saving session...
...saving history...truncating history files...
...completed.
% 
```

##### Using `poetry` non-interactively

You can run python programs using poetry without activating the poetry environment, by using `poetry run {command}`. For example, to run the tests, execute:

```shell
% poetry run python -m unittest
.
--------------------------------
Ran 1 test in 0.000s

OK
```

### PyCharm `poetry` Setup

- Open the project directory in PyCharm.
- Navigate to Preferences > Project > Python Interpreter
- Add a new interpreter. Select "Poetry environment" in the list on the left.

- **If you have _not_ already initialized a `poetry` environment**, specify the following:  
  - Python executable: select the path to an installed python version you wish to use, e.g. 
    `~/.pyenv/versions/3.8.13/bin/python3`
  - Select "install packages from pyproject.toml"
  - Poetry executable: select the path to the poetry installation you have, e.g. 
    `/opt/homebrew/bin/poetry`
  - Click "OK" and wait while the environment builds.

- **If you have already initialized a `poetry` environment**, 
  - ... select the path to it in the "existing environment" dropdown.

- Run the "Python tests for aer" Run/Debug configuration in the PyCharm interface, and check 
  that there are no errors.

### Visual Studio Code `poetry` setup

Prerequisite: run the [Command Line Setup](#command-line-poetry-setup) specified above.

Then:
- Open the project directory in Visual Studio Code
- Install the VSCode plugin recommendations suggested. These include:
  - python
  - python-environment-manager

- **If you have _not_ already initialized a `poetry` environment**:
  - Open a new terminal within Visual Studio Code (Menu: Terminal > New Terminal). 
  - Carry out the [Command Line Setup](#command-line-poetry-setup) specified above.
  - If the python-environment-manager plugin is working correctly, it will suggest adding the environment to the workspace. Do that. If it doesn't, carry out the steps specified below for if you have already initialized a `poetry` environment. 


- **If you have already initialized a `poetry` environment**:
  - Select the `python` option in the vertical bar on the far left of the window. Under the title "PYTHON: ENVIRONMENTS" should be a list of `python` environments. If these do not appear:
    - Ensure the python-environment-manager is installed correctly.
    - Ensure the python-environment-manager is activated.

  - Locate the correct poetry environment. Click the "thumbs up" symbol next to the poetry environment name to "set as active workspace interpreter".
  - Alternatively: Open the command pallete (MacOS: ⇧+⌘+P), enter "Python: Select Interpreter" then hit return. Select the python in the correct environment and hit return.

- Check that the `poetry` environment is correctly set-up. 
  - Open a new terminal within Visual Studio Code (Menu: Terminal > New Terminal). 
  - It should execute something like `source /Users/me/Developer/AER/.venv/bin/activate` before offering you a prompt.
  - If you execute `which python` it should return the path to your python executable in the .venv/ directory.
    - If this does not happen, repeat the steps "If you have already initialized a `poetry` environment" above.
    - If `which python` still does not return the path to your python executable in the .venv/ directory, reach out to someone else who uses Visual Studio Code. If you're on the AER team, post a question in the Slack channel.
  - Run `python -m unittest aer` and ensure that there are no errors.
  

## Set up the Pre-Commit Hooks

We use [`pre-commit`](https://pre-commit.com) to manage pre-commit hooks. 

Pre-commit hooks are programs which run before each git commit and which check that the files to be committed: 
- are correctly formatted and 
- have no obvious coding errors.

Pre-commit hooks are intended to enforce coding guidelines, including the Python style-guide [PEP8](https://peps.python.org/pep-0008/).

The hooks and their settings are specified in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

After cloning the repository and installing the dependencies, you should run:
```zsh
$ pre-commit install
```

to set up the pre-commit hooks.


### Handling Pre-Commit Hook Errors

If your `git commit` fails because of the pre-commit hook, then you should:

1. Run the pre-commit hooks on the files which you have staged, by running the following  command in your terminal: 
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
   **Important: Once you've changed the code, re-stage the files it to Git. This might mean 
   unstaging changes and then adding them again.**
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
