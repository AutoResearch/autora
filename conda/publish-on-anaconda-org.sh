#!/bin/zsh

die() {
  local message="$@"
  >&2 echo -e "${message}"
  exit 1
}

source env.sh  || die ".env file not found. Run: echo ANACONDA_AUTO_RESEARCH_TOKEN=\"\$(anaconda auth --create --name AutoResearchToken --org AutoResearch)\" > .env"
conda config --set anaconda_upload yes
conda build . -c pytorch --output-folder dist/ --token "$ANACONDA_AUTO_RESEARCH_TOKEN"
