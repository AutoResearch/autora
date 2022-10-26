# MCMC-SymReg
Bayesian symbolic regression using mcmc sampling. 

Paper here: https://arxiv.org/abs/1910.08892

## API and Usage

### Codes location

`codes/BSR.py`: API interface of BSR class

`codes/bsr_class.py`: definition of BSR class

`codes/simulations.py`: part of simulation settings in the paper

`codes/funcs.py`: basic sampling functions

### Usage Example

```python
K = 3 # number of trees
MM = 50 # number of iterations
# set hyperparameters alternatively
hyper_params = [{'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1}]
# initialize BSR object
my_bsr = BSR(K,MM)
# train (need to fill in parameters)
# train_X is dataframe with each row a datapoint
# train_y is series with default index
my_bsr.fit(train_X,train_y)
# fit new values
# new_X is dataframe of new data
fitted_y = my_bsr.predict(new_X)
# display fitted trees
my_bsr.model()
# complexity, including complexity of each tree & total
complexity = my_bsr.complexity()
```

>>>>>>> 

## Pdf files

`bsr_paper.pdf`: paper for Bayesian Symbolic Regression

`Symbolic_Regression_Tree_MCMC.pdf`: note for proposed algorithm

