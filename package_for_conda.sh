#!/bin/zsh

# Create a new build directory
tempdir=$(mktemp -d)
echo "new temporary directory: ${tempdir}"
cd $tempdir

echo "Creating a skeleton from PyPI version of AutoRA"
conda skeleton pypi autora

echo "meta.yml file created: "
cat autora/meta.yaml

echo "Building based on the meta.yml file."
conda build autora
