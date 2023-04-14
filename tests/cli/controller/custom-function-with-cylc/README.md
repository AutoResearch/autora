# Example of running the controller CLI under cylc

Requires a conda environment called `autora-cylc` with the following dependencies:
- `autora` 3.0.0a0+
- `cylc-flow`

Run and show output from this directory using
```zsh
cylc install . 
cylc play custom-function-with-cylc
cylc tui custom-function-with-cylc
```
