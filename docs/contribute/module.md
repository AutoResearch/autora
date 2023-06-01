# Contribute an Experimentalist, Experiment Runner, or Theorist

Each experimentalist, experiment runner or theorist is a "child" package based on either
- the [cookiecutter template (recommended)](https://github.com/AutoResearch/autora-template-cookiecutter), or
- the [unguided template](https://github.com/AutoResearch/autora-template).

!!! hint
    The easiest way to contribute a new child package for an experimentalist, experiment runner or theorist,
    start from the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter).

!!! success
    New **synthetic** experiment runners may be submitted as pull requests to the 
    [`autora-synthetic`](https://github.com/autoresearch/autora-synthetic/CONTRIBUTING.md) package, providing they 
    require no additional dependencies. This is meant to simplify small contributions. 
    However, if your contribution requires additional dependencies, you can submit it as a full package following 
    this guide. 

Once your package is working, and you've published it on PyPI, you can **make a pull request** on 
[`autora`](https://github.com/autoresearch/autora) to have it vetted and added to the "parent" package.

The following demonstrates how to add a package published under autora-theorist-example in PyPI in the GitHub 
repository example-contributor/contributor-theorist

## Creating a new child package

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

## Updating a child package

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
