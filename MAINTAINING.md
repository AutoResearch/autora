# Maintainer Guide

## Release Process

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
