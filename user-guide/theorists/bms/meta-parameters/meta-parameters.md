# Meta Parameters

Meta parameters are used to control the search space and the search algorithm. This section provides a basic overview of these parameters along with a description of their effects. 

- **`epochs`**: The number of epochs to run BMS. This corresponds to the total number of equation mutations - one mcmc step for each parallel-tempered equation and one tree swap between a pair of parallel-tempered equations.
- **`prior_par`**: A dictionary of priors for each operation. The keys correspond to operations and the respective values correspond to prior probabilities of those operations. The model comes with a default.  
- **`ts`**: A list of temperature values. The machine scientist creates an equation tree for each of these values. Higher temperature trees are harder to fit, and thus they help prevent overfitting of the model.
