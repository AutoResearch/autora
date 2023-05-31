# Contributor Guide

Contributions to AutoRA are organized into one "parent" and many "child" packages. 

[`autora`](https://github.com/autoresearch/autora) is the "parent" package which end users are expected to install.
It includes vetted "child" packages as optional dependencies which users can choose to install.

Each experimentalist, experiment runner or theorist is a "child" package. 
For details on how to submit child packages for inclusion in `autora`, see
[the module contributor guide here](./module.md).

[`autora-core`](https://github.com/autoresearch/autora-core), is the "core" package which includes fundamental utilities
and building blocks for all the other packages. This is always installed when a user installs `autora` and can be 
a dependency of other "child" packages. For more details, see [the core contributor guide here](./core.md). 

