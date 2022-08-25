# Differentiable Architecture Search

## Meta-Parameters

Meta-parameters are used to control the search space and the search algorithm. DARTS has quite a lot of those parameters. This section provides a basic overview of all parameters along with a description of their effects. 

### General DARTS meta-parameters

- **`num_graph_nodes`**: The number of latent variables used to represent the model.
- **`max_epochs`**: The maximum number of epochs to run DARTS. This corresponds to the total number of architecture updates. These updates affect the architecture weights $\alpha$ indicating the relative contribution of each operation for a given computation step.

### Meta-parameters for the architecture updates
The following parameters affect the updating of the architecture weights $\alpha$:

- **`arch_learning_rate_max`**: The initial (maximum) learning rate for updating the architecture updates. The higher the learning rate, the larger the steps taken to update the architecture weights. The learning rate decays with each epoch.
- **`arch_weight_decay`**: The weight decay for the architecture weights. The higher the weight decay, the more the high architecture weights are pressured to be small.
- **`arch_weight_decay_df`**: An additional weight decay that scales with the number of parameters (degrees of freedom) per operation. The higher this weight decay, the more DARTS will favor operations with few parameters.

### Meta-parameters for the parameter updates
The following parameters affect the updating of the parameters associated with each operation:

- **`param_updates_per_epoch`**: The number of steps taken by the parameter optimizer per epoch. Once the architecture updates are complete, the parameters associated with each operation are updated by a stochastic gradient descent over this number of steps.
- **`param_learning_rate_max`**: The initial (maximum) learning rate for updating the parameters. The higher the learning rate, the larger the steps taken to update the parameters. Note that the learning rate is scheduled to converge over the total number of parameter updates to **`learning_rate_min`**.
- **`param_learning_rate_min`**: The smallest possible learning rate for updating the parameters.
- **`param_momentum`**: The momentum for the architecture updates. The higher the momentum, the more the steps taken to update the architecture weights will be influenced by previous steps.
- **`param_weight_decay`**: The weight decay for the parameters. The higher the weight decay, the more the high parameters of each operation are pressured to be small.

### Meta-parameters for the classifier
The final output of the DARTS model is computed by concatenating all edges in the computation graph into a single vector and then adding a linear classifier. The linear classifier can attach a coefficient to each edge (weighing the contribution of that edge to the final output), and it can add a constant bias term. The following parameters affect the behavior of the classifier:

- **`train_classifier_coefficients`**: If set to `True`, the classifier coefficient of each edge will be trained (otherwise each coefficient is set to `1`, reflecting an equal contribution of each edge to the final output).
- **`train_classifier_bias`**: If set to `True`, the bias term of the classifier will be trained (otherwise the bias term is set to `0`).
