# Conda packaging

This directory includes files for packaging the code for `conda`.
- [`autora/meta.yaml`](./autora/meta.yaml) is the `conda` recipe (configuration file)
- [`package.sh`](./publish-on-anaconda-org.sh) is a script which runs the packaging and outputs the package into the `./dist`

To create and publish the `conda` package:
- update `./autora/meta.yaml` with the new version number and any changed dependencies 
- run `publish-on-anaconda-org.sh`
