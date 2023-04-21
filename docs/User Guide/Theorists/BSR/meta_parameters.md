# Bayesian Symbolic Regression

## Meta-Parameters

Meta-Parameters are used to control the search space and the model configuration. In BSR, they are mainly defined in the theorist constructor (see `bsr.py`). Below is a basic overview of these parameters. Note, there are additional algorithm-irrelevant configurations that can be customized in the constructor; please refer to code documentation for their details.

- `tree_num`: the number of expression trees to use in the linear mixture (final prediction model); also denoted by `K` in BSR.
- `iter_num`: the number of RJ-MCMC steps to execute (note: this can also be understood as the number of `K`-samples to take in the fitting process).
- `val`: the number of validation steps to execute following each iteration.
- `beta`: the hyperparameter that controls growth of a new expression tree. This needs to be < 0, and in general, smaller values of `beta` correspond to deeper expression trees.
