# Conda packaging

This directory includes files for packaging the code for `conda`.
- [`autora/meta.yaml`](./autora/meta.yaml) is the `conda` recipe (configuration file)
- [`publish-on-anaconda-org.sh`](./publish-on-anaconda-org.sh) is a script which runs the packaging and outputs the package into the `./dist` directory.

To create and publish the `conda` package:
- update `./autora/meta.yaml` with the new version number and any changed dependencies. Commit these changes.
- 🐛 Bugfix: While poetry >=1.2 is not available on anaconda.org, delete the [tool.poetry.group...] parts of pyproject.toml. These are recognized as incorrect Poetry configuration in poetry 1.1 and below, and will cause the build to fail. Don't commit these changes.  
- run `publish-on-anaconda-org.sh`
