# Automated Research Assistant
Automated Research Assistant (AutoRA) is an open source AI-based system for automating each aspect of empirical research in the behavioral sciences, from the construction of a scientific hypothesis to conducting novel experiments. The documentation is here: [https://autoresearch.github.io/autora/](https://autoresearch.github.io/autora/)

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

## Development Practices

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

# Documentation

## Commenting code

To help users understand code better, and to make the documentation generation automatic, we have some standards for documenting code. The comments, docstrings, and the structure of the code itself are meant to make life easier for the reader. 
- If something important isn't _obvious_ from the code, then it should be _made_ obvious with a comment. 
- Conversely, if something _is_ obvious, then it doesn't need a comment.

These standards are inspired by John Ousterhout. *A Philosophy of Software Design.* Yaknyam Press, 2021. Chapter 12 – 14.

### Every public function, class and method has documentation

We include docstrings for all public functions, classes, and methods. These docstrings are meant to give a concise, high-level overview of **why** the function exists, **what** it is trying to do, and what is **important** about the code. (Details about **how** the code works are often better placed in detailed comments within the code.)

Every function, class or method has a one-line **high-level description** which clarifies its intent.   

The **meaning** and **type** of all the input and output parameters should be described.

There should be **examples** of how to use the function, class or method, with expected outputs, formatted as ["doctests"](https://docs.python.org/3/library/doctest.html). These should include normal cases for the function, but also include cases where it behaves unexpectedly or fails. 

We follow the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), as these are supported by the online documentation tool we use (see [Online Documentation](#online-documentation)).

A well documented function looks something like this:
```python
def first_order_linear(
    x: Union[float, np.ndarray], c: float, m: float
) -> Union[float, np.ndarray]:
    """
    Evaluate a first order linear model of the form y = m x + c.

    Arguments:
        x: input location(s) on the x-axis
        c: y-intercept of the linear model
        m: gradient of the linear model

    Returns:
        y: result y = m x + c, the same shape and type as x

    Examples:
        >>> first_order_linear(0. , 1. , 0. )
        1.0
        >>> first_order_linear(np.array([-1. , 0. , 1. ]), c=1.0, m=2.0)
        array([-1.,  1.,  3.])
    """
    y = m * x + c
    return y
```

*Pro-Tip: Write the docstring for your new high-level object before starting on the code. In particular, writing examples of how you expect it should be used can help clarify the right level of abstraction.*

## Online Documentation

Online Documentation is automatically generated using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) based on docstrings in files in the `autora/` directory. 

### Commands

Build and serve the documentation using the following commands:

* `poetry run mkdocs serve` - Start the live-reloading docs server.
* `poetry run mkdocs build` - Build the documentation site.
* `poetry run mkdocs gh-deploy` - Build the documentation and serve at https://AutoResearch.github.io/AutoRA/
* `poetry run mkdocs -h` - Print help message and exit.

### Documentation layout
```
mkdocs.yml    # The configuration file for the documentation.
docs/         # Directory for static pages to be included in the documentation.
    index.md  # The documentation homepage.
    ...       # Other markdown pages, images and other files.
autora/          # The directory containing the source code.
```
# Release Process

The release process is automated using GitHub Actions. 

- Before you start, ensure that the tokens are up-to-date. If in doubt, try to create and publish a new release 
  candidate version of the package first. The tokens are stored as "organization secrets" enabled for the autora 
  repository, and are called:
  - PYPI_TOKEN: a token from pypi.org with upload permissions on the AutoResearch/AutoRA project.
  - ANACONDA_TOKEN: a token from anaconda.org with the following scopes on the AutoResearch organization: `repos conda
    api:read api:write`. Current token expires on 2023-03-01.
- Update [conda recipe](./conda/autora/meta.yaml): 
    - dependencies, so that it matches [pyproject.toml](pyproject.toml).
    - imports for testing – all modules should be listed.
- Trigger a new release from GitHub. 
  - Navigate to the repository's code tab at https://github.com/autoresearch/autora,
  - Click "Releases",
  - Click "Draft a new release",
  - In the "Choose a tag" field, type the new semantic release number using the [PEP440 syntax](https://peps.python.
    org/pep-0440/). The version number should be prefixed with a "v". 
    e.g. "v1.2.3" for a standard release, "v1.2.3a4" for an alpha release, "v1.2.3b5" for a beta release, 
    "v1.2.3rc6" for a release candidate, and then click "Create new tag on publish". 
  - Leave "Release title" empty.
  - Click on "Generate Release notes". Check that the release notes match with the version number you have chosen – 
    breaking changes require a new major version number, e.g. v2.0.0, new features a minor version number, e.g. 
    v1.3.0 and fixes a bugfix number v1.2.4. If necessary, modify the version number you've chosen to be consistent 
    with the content of the release.
  - Select whether this is a pre-release or a new "latest" release. It's a "pre-release" if there's an alpha, 
    beta, or release candidate number in the tag name, otherwise it's a new "latest" release.
  - Click on "Publish release"
- GitHub actions will run to create and publish the PyPI and Anaconda packages, and publish the documentation. Check in 
  GitHub actions whether they run without errors and fix any errors which occur.

# How to add new packages

This demonstrates how to add a package published under autora-theorist-example in pyPI in the GitHub repository 
example-contributor/contributor-theorist

## Add the package as optional dependency
In the `pyorject.toml` file add an optional dependency for the package in the [project.optional-dependencies] section:
```toml
example-theorist = ["autora-theorits-example"]
```
Add the example-theorist to be part of the all-theorists dependency:
```toml
all-theorists = [
    ...
    "autora[example-theorist]",
    ...
]
```

## Import documentation from the package repository
Import the documentation in the `mkdocs.yml` file:
```yml
- User Guide:
  - Theorists:
    - Overview: 'theorist/overview.md'
    ...
    - Example Theorist: '!import https://github.com/example-contributor/contributor-theorist/?branch=main&extra_imports=["mkdocs/base.yml"]'
    ...
```

# How to Develop

Install this in an environment using your chosen package manager. 

## Using `virtualenv`

Install:
- python: https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Create a new virtual environment:
```shell
virtualenv venv
```

Activate it:
```shell
source venv/bin/activate
```

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

Build the package using:
```shell
python -m build
```

Publish the package to PyPI using `twine`:
```shell
twine upload dist/* 
```
