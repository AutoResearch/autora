# Conda packaging

This directory includes files for packaging the code for `conda`.
- [`autora/meta.yaml`](./autora/meta.yaml) is the `conda` recipe (configuration file)
- [`package.sh`](./publish-on-anaconda-org.sh) is a script which runs the packaging and outputs the package into the `./dist`

To create the `conda` package, run `package.sh`
