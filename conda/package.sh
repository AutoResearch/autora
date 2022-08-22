#!/bin/zsh
conda config --set anaconda_upload no
package=$(conda build . -c pytorch --output-folder dist/ --output)
anaconda upload -u AutoResearch "${package}"
