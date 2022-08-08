#!/bin/zsh

echo "Building based on the meta.yml file."
conda build . -c pytorch --output-folder dist/
