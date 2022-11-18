#!/bin/zsh

conda config --set anaconda_upload yes
TOKEN=$(anaconda auth --create --org AutoResearch --name "$(uname -n)-$(date +%s)" --max-age 3600 --scopes anaconda_upload)
conda build ./recipe -c pytorch --output-folder dist/ --token "$TOKEN"
