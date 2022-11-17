#!/bin/zsh

conda config --set anaconda_upload yes
TOKEN=$(anaconda auth --create --org AutoResearch --name "$(uname -n)-$(date +%s)" --max-age 3600  || die "Token could not be created.")
conda build . -c pytorch --output-folder dist/ --token "$TOKEN"
