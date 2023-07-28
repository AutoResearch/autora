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


## Implement Your Module

After setting up your repository and linking it to your GitHub account, you can start implementing your module.

### Implement Your Code

You may implement your code in the ``init.py`` located in the respective feature folder in ``src/autora``.

Please refer to the following guides on implementing
- [theorists](theorist.md)
- [experimentalists](experimentalist.md)
- [experiment runners](experiment-runner.md)

If the feature you seek to implement does not fit in any of these categories, then 
you can create folders for new categories. If you are unsure how to proceed, you are always welcome 
to ask for help in the [AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-contributions).

### Add Tests (Optional)

It is highly encouraged to add unit tests to ensure your code is working as intended. These can be [doctests](https://docs.python.org/3/library/doctest.html) or test cases in `tests/test_your_contribution_name.py`.
For example, if you are implementing a sampler experimentalist, you may rename and modify the 
``tests/test_experimentalist_sampler_example.py``.

*Note: Tests are required for your module to become part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to implement tests, 
you will still be able to install your package separately, in addition to `autora`.* 

### Add Documentation (Optional)

It is highly encouraged that you add documentation of your package in `docs/index.md`. You can also add new or delete unnecessary pages 
in the `docs` folder. However you structure your documentation, be sure that structure is reflected in the `mkdocs.yml` file.

You are also encouraged to describe basic usage of your module in the 
python notebook ``Basic Usage.ipynb`` in the `docs` folder. Finally you can outline the basic setup of your module in 
the `docs/quickstart.md` file.

*Note: Documentation is required for your module to become part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to write
documentation, you will still be able to install your package separately, in addition to `autora`.*

### Add Dependencies

In the `pyproject.toml` file, add the new dependencies under `dependencies`.

Install the added dependencies
```shell
pip install -e ".[dev]"
```

## Publish Your Module

There are several ways to publish your package, depending on how you set up your repository.

### Publishing Via GitHub Actions
If you used the **cookiecutter template** with the advanced setup, and uploaded your repository to github.com, then you can use Github Actions to automatically publish your package to PyPI or Conda. 

Note, if your repository is part of the [AutoResearch Organization](https://github.com/AutoResearch) you can skip the step below for creating a new secret in your repository.

1. Add an API token to the GitHub Secrets
    - Create a [PyPI account](https://pypi.org/) if you don't have one already.
    - Once you have an account, generate an API token for your account.
    - In your GitHub repository, go to `Settings`.
    - Under `Secrets and variables` in the left-hand menu, select `Actions`. 
    - Create a new secret named `PYPI_API_TOKEN` and paste in your PyPI API token as the value.

2. Create a new release
    - Follow the steps outlined in the [GitHub documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) for creating a new release. 
    - Once you create a new release, the GitHub Action will automatically trigger, and your package will be built and published to PyPI using the provided API token.

### Manually Publishing
If you used the **unguided template**, or you want to manually publish your package, you can follow [step 7 in this guide](https://github.com/AutoResearch/autora-template#step-7-publish-your-package).

Once you've published your module, you should take some time to celebrate and announce your contribution in the 
[AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-announcements).

## Incorporate Your Module Into The AutoRA Parent Package

Once your package is working and published, you can **make a pull request** on [`autora`](https://github.com/autoresearch/autora) to have it vetted and added to the "parent" package. Note, if you are not a member of the AutoResearch organization on GitHub, you will need to create a fork of the repository for the parent package and submit your pull request via that fork. If you are a member, you can create a pull request from a branch created directly from the parent package repository. Steps for creating a new branch to add your module are specified below.

!!! success
    In order for your package to be included in the parent package, it must meet the following criteria:
    - have basic documentation in ``docs/index.md``
    - have a basic python notebook exposing how to use the module in ``docs/Basic Usage.ipynb``
    - have basic tests in ``tests/``
    - be published via PyPI or Conda
    - be compatible with the current version of the parent package
    - follow standard python coding guidelines including PEP8

The following demonstrates how to add a package published under `autora-theorist-example` in PyPI in the GitHub 
repository `example-contributor/contributor-theorist`.

### Install The Parent Package In Development Mode

!!! success
    We recommend setting up your development environment using a manager like `venv`, which creates isolated python 
    environments. Other environment managers, such as 
    [virtualenv](https://virtualenv.pypa.io/en/latest/),
    [pipenv](https://pipenv.pypa.io/en/latest/),
    [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), 
    [hatch](https://hatch.pypa.io/latest/), 
    [poetry](https://python-poetry.org), 
    are available and will likely work, but will have syntax different to that shown here. 

Run the following command to create a new virtual environment in the `.venv` directory

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

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

Check that the documentation builds correctly by running:
```shell
mkdocs serve
```

... then viewing the documentation using the link in your terminal.

### Create A New Branch Of The Parent Package

Once you've successfully installed the parent package in development mode, you can begin the process of adding your contribution by creating a new branch off of the `main` branch. You should name your branch according to the name of your contribution. In the example we're using here, the branch would be called `feat/contributor-theorist`. After creating your branch, you can start making the modifications specified below. 

### Add The Package As An Optional Dependency

In the `pyorject.toml` file add an optional dependency for the package in the `[project.optional-dependencies]` section:

```toml
theorist-example = ["autora-theorist-example==1.0.0"]
```

!!! success
    Ensure you include the version number.

Add the `theorist-example` to be part of the `all-theorists` dependency:
```toml
all-theorists = [
    ...
    "autora[theorist-example]",
    ...
]
```

Update the environment:

```shell
pip install -U -e ".[dev]"
```

... and check that your package is still importable and works as expected.

### Import Documentation From The Package Repository

Import the documentation in the `mkdocs.yml` file:
```yml
plugins:
  multirepo:
    nav_repos:
      ...
      - name: theorist-example
        import_url: "https://github.com/example-contributor/contributor-theorist/?branch=v1.0.0"
        imports: [ "src/" ]
  ...
  mkdocstrings:
    handlers:
      python:
        paths: [
          ...,
          "./temp_dir/theorist-example/src/"
        ]
...
- User Guide:
  - Theorists:
    - Home: 'theorist/index.md'
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

## Updating Your Module

!!! warning
    Please note, that packages need to be vetted each time they are updated.

In the `[project.optional-dependencies]` section of the `pyproject.toml` file, update the version number:
```toml
theorist-example = ["autora-theorist-example==1.1.0"]
```

Also update the version number in the `mkdocs.yml`: 
```yml
plugins:
  multirepo:
    nav_repos:
      ...
      - name: theorist-example
        import_url: "https://github.com/example-contributor/contributor-theorist/?branch=v1.1.0"
        imports: [ "src/" ]
...
- User Guide:
  - Theorists:
    ...
    - Example Theorist: '!import https://github.com/example-contributor/contributor-theorist/?branch=v1.1.0&extra_imports=["mkdocs/base.yml"]'
    ...
```

Next, update the environment:
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

Note, whenever you update and release a new version of your module, you will need to add the new version number in the places described above and create a new PR to have it included in `autora`.



