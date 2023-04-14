running on command line using default next step:
```zsh
python -m autora.controller controller.yml history/. --verbose --debug 
```

running a particular next step:
```zsh
python -m autora.controller controller.yml history/. --step-name=experimentalist --verbose --debug
python -m autora.controller controller.yml history/. --step-name=experiment_runner --verbose --debug
python -m autora.controller controller.yml history/. --step-name=theorist --verbose --debug 
```
