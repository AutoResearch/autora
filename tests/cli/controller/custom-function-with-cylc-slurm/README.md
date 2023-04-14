# Example of running the controller CLI under cylc

Requires a conda environment called `autora-cylc` with the following dependencies:
- `autora` 3.0.0a0+
- `cylc-flow`

```bash
conda activate /users/jholla10/anaconda/autora-cylc
```

Run and show output from this directory using
```bash
cylc install . 
cylc play custom-function-with-cylc-slurm
cylc tui custom-function-with-cylc-slurm
```
