# Contributions To Core Packages

Core contributions are changes to AutoRA which aren't experimentalists, (synthetic) experiment runners, or theorists. 
The primary purpose of the core is to provide utilities for:

- describing experiments (in the [`autora-core` package](https://github.com/autoresearch/autora-core))
- handle workflows for automated experiments
  (currently in the [`autora-workflow` package](https://github.com/autoresearch/autora-workflow))
- run synthetic experiments (currently in the [`autora-synthetic` package](https://github.com/autoresearch/autora-synthetic). Synthetic experiment runners may be submitted as pull requests to the 
    [`autora-synthetic`](https://github.com/AutoResearch/autora-synthetic/blob/main/CONTRIBUTING.md) package, providing they 
    require no additional dependencies. However, if your contribution requires additional dependencies, you can submit it as a full package following 
    the [module contribution guide](modules/index.md).

## How to Contribute

In order to make a contribution to autora core, you may start by cloning the repository:
```bash
git clone https://github.com/AutoResearch/autora-core.git
```

You should then follow the [Development Setup Guide](https://autoresearch.github.io/autora/contribute/setup/), to set up your environment. 

Suggested changes to the core should be submitted as follows, depending on their content:

- For fixes or new features closely associated with existing core functionality: pull request to the existing 
  core package
- For new features which don't fit into the current module structure, or which are experimental and could lead to 
  instability for users: as new namespace packages.

!!! success
    Reach out to the core team about new core contributions to discuss how best to incorporate them by posting your 
    idea on the [discussions page](https://github.com/orgs/AutoResearch/discussions/categories/ideas).

## Development Requirements

Core packages should as a minimum:

- Follow standard python coding guidelines including PEP8
- Run under all minor versions of python (e.g. 3.8, 3.9) allowed in 
  [`autora-core`](https://github.com/autoresearch/autora-core)
- Be compatible with all current AutoRA packages
- Have comprehensive test suites
- Use the linters and checkers defined in the `autora-core` 
  [.pre-commit-config.yaml](https://github.com/AutoResearch/autora-core/blob/main/.pre-commit-config.yaml) (see [pre-commit hooks](pre-commit-hooks.md))
