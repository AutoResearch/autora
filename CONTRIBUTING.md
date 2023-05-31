# Contributor Guide

Contributions to AutoRA are organized into one "parent" and many "child" packages. 

[`autora`](https://github.com/autoresearch/autora) is the "parent" package which end users are expected to install.
It includes all vetted "child" packages as optional dependencies which users can choose to install. 

[`autora-core`](https://github.com/autoresearch/autora-core), is the "core" package which includes fundamental utilities
and building blocks for all the other packages. This is always installed when a user installs `autora` and can be 
a dependency of other "child" packages.

Each "child" package adds a new experimentalist, experiment runner or theorist. These are based on the 
[cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter).

## Contributing a new child package

> 🎁 To contribute a new child package,
  start from the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter).

Once your package is working, and you've published it on PyPI, you can **make a pull request** on 
[`autora`](https://github.com/autoresearch/autora) to have it vetted and added to the "parent" package.

The following demonstrates how to add a package published under autora-theorist-example in pyPI in the GitHub 
repository example-contributor/contributor-theorist

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

> ✅ Ensure you include the version number.

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

> ✅ Ensure you include the version number in the `!import` string after `?branch=`. Ensure that the commit you want 
> to submit has a tag with the correct version number in the correct format.

Check that the documentation builds correctly by running:
```shell
mkdocs serve
```

... then view the documentation using the link in your terminal. Check that your new documentation is included in 
the right place and renders correctly.

## Updating a child package

> ❗️Please note, that packages from external contributors need to be vetted each time they are updated.

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
