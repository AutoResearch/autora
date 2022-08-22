#!/bin/zsh
conda config --set anaconda_upload yes
TOKEN=$(anaconda auth --create --name AutoResearchToken --org AutoResearch)
conda build . -c pytorch --output-folder dist/ --token "$TOKEN"
