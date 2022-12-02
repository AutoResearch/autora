import numpy as np
import matplotlib.pyplot as plt

from autora.skl.darts import DARTSRegressor

# generate the synthetic data: y = exp(x)
x = np.expand_dims(np.linspace(start=-1, stop=1, num=500), 1)
y = np.exp(x)

# define the primitives
primitives = [
    "none",
    "add",
    "subtract",
    "logistic",
    "exp",
    "relu",
    "cos",
    "sin",
    "tanh",
]

# train a DARTS regressor to recover the function
regressor = DARTSRegressor(
    num_graph_nodes=4,
    param_updates_per_epoch=100,
    max_epochs=300,
    arch_updates_per_epoch=1,
    param_weight_decay=3e-4,
    arch_weight_decay_df=0.001,
    arch_weight_decay=1e-4,
    arch_learning_rate_max=0.3,
    param_learning_rate_max=0.0025,
    param_learning_rate_min=0.01,
    param_momentum=0.90,
    primitives=primitives,
    train_classifier_bias=False,
    train_classifier_coefficients=False,
)

regressor.fit(x, y)

# resample the architecture parameters
regressor.set_params(
    sampling_strategy="max",
    param_updates_for_sampled_model=100,
)
regressor.fit(x, y)

# plot the ground truth against the model's prediction
y_predict = regressor.predict(x)
# print the model
print(regressor.model_repr())

# visualize the model
graph = regressor.visualize_model()
graph.render(view=True)
