# Contributor Guide

The AutoRA project is a collection of packages which together form a framework for closed-loop empirical research.
We invite contributions to all parts of the project, including the [core package](core.md), the 
[parent package](core.md), and the [modules](module.md). Below is a brief overview of the
project structure, along with pointers to more detailed contribution guides for each part of the project.

## Project Structure

Contributions to AutoRA are organized into one "parent" and many "child" packages. 
Child packages are generally maintained by individual contributors. The parent package, along with some other 
*core* packages, is maintained by [Autonomous Empirical Research Group](https://musslick.github.io/AER_website/Team.html), 
as well as external contributors.

![image](../img/package_overview.png)

[`autora`](https://github.com/autoresearch/autora) is the "parent" package which end users are expected to install. The
parent depends on core packages, such as [`autora-core`](https://github.com/autoresearch/autora-core), 
[`autora-workflow`](https://github.com/autoresearch/autora-workflow), and
[`autora-synthetic`](https://github.com/autoresearch/autora-synthetic). It also includes vetted modules (child packages) as optional dependencies which users can choose 
to install. 

You may contribute to any of the core packages or develop your own module as a stand-alone package (see below).    


### Module Contributions

Each theorist, experimentalist, or experiment runner is a child package. Child packages are owned and maintained by you, the contributor, which provides several advantages:
- *Easy setup*: We provide simple [templates](module.md) for child packages, which you can use to get started quickly
- *Independence*: You can develop and maintain your package independently of other child packages (and thereby avoid dependency conflicts)
- *Ownership*: You can publish your package on PyPI, use it in other projects, and get credit for its use. 

For details on how to submit child packages 
for inclusion in `autora`, see
[the module contributor guide](./module.md). 

Feel free to post questions and feedback regarding core contributions on the 
[AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-contributions).

### Core Contributions

The following packages are considered "core" packages, and are actively maintained by the
[Autonomous Empirical Research Group](https://musslick.github.io/AER_website/Team.html):

- **autora** [`https://github.com/autoresearch/autora`](https://github.com/autoresearch/autora): The parent package use the one that users install, e.g., via `pip install autora`. The package determines which modules (child packages) are included and maintains the general documentation.


- **autora-core** [`https://github.com/autoresearch/autora-core`](https://github.com/autoresearch/autora-core) This package  includes fundamental utilities
and building blocks for all the other packages. This is always installed when a user installs `autora` and can be 
a dependency of other "child" packages.   


- **autora-workflow** [`https://github.com/autoresearch/autora-workflow`](https://github.com/autoresearch/autora-workflow): The workflow packages includes basic utilities for managing the workflow of closed-loop research processes, e.g., coordinating workflows between the theorists, experimentalists, and experiment runners. This package is expected to be merged into autora-core.


- **autora-synthetic** [`https://github.com/autoresearch/autora-synthetic`](https://github.com/autoresearch/autora-synthetic): This package a number of ground-truth models from different scientific disciplines that can be used for benchmarking automated scientific discovery. If you seek to contribute a scientific model, please see the [core contributor guide](core.md) for details.   


We welcome contributions to
these packages in the form of pull requests, bug reports, and feature requests. For more details, see the
[core contributor guide](core.md). 

For core contributions, it is possible to set up your python environment in many different ways. 
One setup which works for us is described in [the setup guide](./setup.md). 

We welcome questions and feedback regarding core contributions on the 
[AutoRA forum](https://github.com/orgs/AutoResearch/discussions/categories/module-contributions).

!!! hint
    If you would like to become actively involved in the development and maintenance of core AutoRA packages, 
    we welcome you to join the [Autonomous Empirical Research Group](https://musslick.github.io/AER_website/Team.html).

