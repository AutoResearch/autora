import matplotlib.pyplot as plt
import numpy as np

from autora.cycle import Cycle
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler import grid_pool, poppernet_pool
from autora.experimentalist.sampler import nearest_values_sampler
from autora.skl.bms import BMSRegressor
from autora.variable import Variable, VariableCollection
from autora.experiment_runner.tinkerforge.experiment_client import run_experiment

# meta parameters
ground_truth_resolution = 1000
samples_per_cycle = 6
value_range = (0, 3.5)
allowed_values = np.linspace(value_range[0], value_range[1], ground_truth_resolution)


# define ground truth
def run_tinkerforge_experiment(x, study_metadata):

    xs = x.copy()

    if xs.ndim == 1:
        xs = xs.reshape(-1, 1)

    # get list of IVs
    IVs = list()
    for idx, iv in enumerate(study_metadata.independent_variables):
        IVs.append(iv.name)
        xs[:, idx] = xs[:, idx] * iv.rescale

    # get list of DVs
    DVs = list()
    for dv in study_metadata.dependent_variables:
        DVs.append(dv.name)

    data = run_experiment(xs, IVs, DVs)

    y = np.empty((len(xs), len(DVs)))
    for i, dv in enumerate(DVs):
        y[:, i] = data[dv]

    return y

# define variables
study_metadata = VariableCollection(
    independent_variables=[
        Variable(name="source_voltage",
                 allowed_values=allowed_values,
                 value_range=value_range,
                 units="V",
                 variable_label="Source Voltage",
                 rescale = 1000)
    ],
    dependent_variables=[Variable(name="current0",
                                  value_range=(-5000, 5000),
                                  units="mA",
                                  variable_label="Current")
    ]
)

# define experiment platform
def get_tinkerforge_experiment_runner(metadata):
    def runner(xs):
        return run_tinkerforge_experiment(xs, metadata)
    return runner

tinkerforge_experiment_runner = get_tinkerforge_experiment_runner(metadata=study_metadata)

# Initialize the experimentalist
seed_experimentalist = Pipeline(
    [
        ("grid_pool", grid_pool),
        ("nearest_values_sampler", nearest_values_sampler),
    ],
    {
        "grid_pool": {"ivs": study_metadata.independent_variables},
        "nearest_values_sampler": {
            "allowed_values": np.linspace(
                value_range[0], value_range[1], samples_per_cycle
            ),
            "n": samples_per_cycle,
        },
    },
)

# define theorist
bms_theorist = BMSRegressor(epochs=500)

# define seed cycle
# we will use this cycle to collect initial data and initialize the BMS model
seed_cycle = Cycle(
    metadata=study_metadata,
    theorist=bms_theorist,
    experimentalist=seed_experimentalist,
    experiment_runner=tinkerforge_experiment_runner,
)

# run seed cycle
seed_cycle.run(num_cycles=1)

# retrieve model
seed_model = seed_cycle.data.theories[0].model_
seed_x = seed_cycle.data.conditions[0]
seed_y = seed_cycle.data.observations[0][:, 1]

# now we define the poppernet experimentalist which takes into account
# the seed data and the seed model
popper_experimentalist = Pipeline(
    [
        ("popper_pool", poppernet_pool),
        ("nearest_values_sampler", nearest_values_sampler),
    ],
    {
        "popper_pool": {
            "metadata": study_metadata,
            "model": seed_model,
            "x_train": seed_x,
            "y_train": seed_y,
            "n": samples_per_cycle,
            "plot": True,
        },
        "nearest_values_sampler": {
            "allowed_values": allowed_values,
            "n": samples_per_cycle,
        },
    },
)

# running a new cycle taking into account the seed data and model
# TODO: need to find a way to incorporate the seed data into the cycle
cycle = Cycle(
    metadata=study_metadata,
    theorist=bms_theorist,
    experimentalist=popper_experimentalist,
    experiment_runner=tinkerforge_experiment_runner,
)
cycle.run(num_cycles=1)


# plot output of architecture search
all_obs = np.row_stack(seed_cycle.data.observations)
x_obs, y_obs = all_obs[:, 0], all_obs[:, 1]
plt.scatter(x_obs, y_obs, s=10, label="seed data")

x_pred = np.array(study_metadata.independent_variables[0].allowed_values).reshape(
    ground_truth_resolution, 1
)
y_pred_seed = seed_cycle.data.theories[0].predict(x_pred)
y_pred_final = cycle.data.theories[0].predict(x_pred)
plt.plot(x_pred, y_pred_seed, color="blue", label="seed model")
plt.plot(x_pred, y_pred_final, color="red", label="final model")
plt.legend()
plt.show()
