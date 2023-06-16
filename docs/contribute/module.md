# Module Contributions

Theorists, experimentalists, experiment runners and other novel functionalities are implemented as "child" packages. 
They are based on either

- the [cookiecutter template (recommended)](https://github.com/AutoResearch/autora-template-cookiecutter), or
- the [unguided template](https://github.com/AutoResearch/autora-template).

!!! hint
    The easiest way to contribute a new child package for an experimentalist, experiment runner or theorist,
    start from the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter).

!!! success
    New **synthetic** experiment runners may be submitted as pull requests to the 
    [`autora-synthetic`](https://github.com/AutoResearch/autora-synthetic/blob/main/CONTRIBUTING.md) package, providing they 
    require no additional dependencies. This is meant to simplify small contributions. 
    However, if your contribution requires additional dependencies, you can submit it as a full package following 
    this guide. 


## Implementing your module

After setting up your repository and linking it to your GitHub account, you can start implementing your module.

### Step 1: Implement Your Code

You may implement your code in the ``init.py`` located in the respective feature folder in ``src/autora``.

Please refer to the following guides on implementing
- [theorists](theorist-module.md)
- [experimentalists](experimentalist-module.md)
- [experiment runners](experiment-runner-module.md)

If the feature you seek to implement does not fit in any of these categories, then 
you can create folders for new categories. If you are unsure how to proceed, you are always welcome 
to ask for help in the [AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-contributions).

### Step 2 (Optional): Add Tests

It is highly encouraged to add unit tests to ensure your code is working as intended. These can be [doctests](https://docs.python.org/3/library/doctest.html) or test cases in `tests/test_your_contribution_name.py`.
For example, if you are implementing a sampler experimentalist, you may rename and modify the 
``tests/test_experimentalist_sampler_example.py``.

*Note: Tests are required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to implement tests, 
you will still be able to install your package separately, in addition to autora.* 

### Step 3 (Optional): Add Documentation

It is highly encouraged that you add documentation of your package in your `docs/index.md`. You can also add new pages 
in the `docs` folder. Update the `mkdocs.yml` file to reflect structure of the documentation. For example, you can add 
new pages or delete pages that you deleted from the `docs` folder.

You are also encouraged to describe basic usage of your theorist in the 
python notebook ``Basic Usage.ipynb`` in the `docs` folder. Finally you can outline the basic setup of your theorist in 
the `docs/quickstart.md` file.

*Note: Docmentation is required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to write
documentation, you will still be able to install your package separately, in addition to autora.*

### Step 4: Add Dependencies

In pyproject.toml add the new dependencies under `dependencies`

Install the added dependencies
```shell
pip install -e ".[dev]"
```

## Publishing your module

There are several ways to publish your package, depending on how you set up your repository.

- If you used the **cookiecutter template** with the advanced setup, and uploaded your repository to 
github.com, then you can use Github Actions to automatically publish your package to PyPI or Conda. 

- If you used the **unguided template**, or you want to manually publish your package, you can follow [step 7 in this guide](https://github.com/AutoResearch/autora-template).

Once you've published your module, you should take some time to celebrate and announce your contribution in the 
[AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-announcements).

## Incorporating your module into the `autora` parent package

Once your package is working and published, you can **make a pull request** on 
[`autora`](https://github.com/autoresearch/autora) to have it vetted and added to the "parent" package.
The following demonstrates how to add a package published under autora-theorist-example in PyPI in the GitHub 
repository example-contributor/contributor-theorist

!!! success
    In order for your package to be included in the parent package, it must
    - include basic documentation in ``docs/index.md``
    - include a basic python notebook exposing how to use the module in ``docs/Basic Usage.ipynb``
    - include basic tests in ``tests/``
    - be published via PyPI or Conda
    - be compatible with the current version of the parent package
    - follow standard python coding guidelines including PEP8

### Install the "parent" package in development mode

Install this in an environment using your chosen package manager. In this example, we use pip and virtualenv.

First, install:

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

Check that the documentation builds correctly by running:
```shell
mkdocs serve
```

... then viewing the documentation using the link in your terminal.


### Add the package as optional dependency
In the `pyorject.toml` file add an optional dependency for the package in the `[project.optional-dependencies]` section:

```toml
example-theorist = ["autora-theorist-example==1.0.0"]
```

!!! success
    Ensure you include the version number.

Add the example-theorist to be part of the all-theorists dependency:
```toml
all-theorists = [
    ...
    "autora[example-theorist]",
    ...
]
```

Update the environment:

```shell
pip install -U -e ".[dev]"
```

... and check that your package is still importable and works as expected.

### Import documentation from the package repository
Import the documentation in the `mkdocs.yml` file:
```yml
- User Guide:
  - Theorists:
    - Overview: 'theorist/overview.md'
    ...
    - Example Theorist: '!import https://github.com/example-contributor/contributor-theorist/?branch=v1.0.0&extra_imports=["mkdocs/base.yml"]'
    ...
```

!!! success
    Ensure you include the version number in the `!import` string after `?branch=`. Ensure that the commit you want 
    to submit has a tag with the correct version number in the correct format.

Check that the documentation builds correctly by running:
```shell
mkdocs serve
```

... then view the documentation using the link in your terminal. Check that your new documentation is included in 
the right place and renders correctly.

## Updating your module

!!! warning
    Please note, that packages need to be vetted each time they are updated.

Update the version number in the  `pyproject.toml` file, in the [project.optional-dependencies] 
section:
```toml
example-theorist = ["autora-theorist-example==1.1.0"]
```

Update the version number in the `mkdocs.yml`: 
```yml
- User Guide:
  - Theorists:
    ...
    - Example Theorist: '!import https://github.com/example-contributor/contributor-theorist/?branch=v1.1.0&extra_imports=["mkdocs/base.yml"]'
    ...
```

Update the environment:
```shell
pip install -U -e ".[dev]"
```

... and check that your package is still importable and works as expected.

Check that the documentation builds correctly by running:
```shell
mkdocs serve
```

... then view the documentation using the link in your terminal. Check that your new documentation is included in 
the right place and renders correctly.


Once everything is working locally, make a new PR on [github.com](https://github.com/autoresearch/autora) with your 
changes. Include: 

- a description of the changes to the package, and 
- a link to your release notes. 

Request a review from someone in the core team and wait for their feedback!



