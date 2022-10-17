# Bayesian Machine Scientist

## Meta-Parameters

Meta-parameters are used to control the search space and the search algorithm. This section provides a basic overview of all parameters along with a description of their effects. 

- **`epochs`**: The number of epochs to run BMS. This corresponds to the total number of equation mutations - one mcmc step for each parallel-tempered equation and one tree swap between a pair of parallel-tempered equations
- **`prior_par`**: A dictionary of priors for each operation. The keys correspond to operations and the values correspond to a measure of the prior probability of that operation occurring. The model comes with a default.  
- **`ts`**: A list of temperature values. The machine scientist creates an equation tree for each temperature. Higher temperature trees are harder to fit, and thus they help to avoid overfitting the model.
