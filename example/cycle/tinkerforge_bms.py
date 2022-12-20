import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from functools import partial
import numpy as np

from autora.cycle import Cycle
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler import grid_pool, poppernet_pool
from autora.experimentalist.sampler import nearest_values_sampler
from autora.skl.bms import BMSRegressor
from autora.variable import Variable, VariableCollection
from autora.experiment_runner.tinkerforge.experiment_client import run_experiment
from typing import List

# meta parameters
# great run with 10, 10, 500

ground_truth_resolution = 100
samples_per_cycle = 10 # 6
num_popper_cycles = 5 # 1
bms_epochs = 500 # 500
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

numIVs = len(study_metadata.independent_variables)
numDVs = len(study_metadata.dependent_variables)

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
bms_theorist = BMSRegressor(epochs=bms_epochs)

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

observations_list = list()
theorist_list = list()
conditions_list = list()

for cycle in range(num_popper_cycles):
    if cycle == 0:
        # retrieve model
        model_seed = seed_cycle.data.theories[0].model_
        x_seed = seed_cycle.data.conditions[0]
        y_seed = seed_cycle.data.observations[0][:, 1]
        theory_seed = seed_cycle.data.theories[0]

    observations_list.append(np.column_stack([x_seed, y_seed]))
    theorist_list.append(theory_seed)
    conditions_list.append(x_seed)

    popper_x_seed = np.vstack(observations_list)[:,:numIVs]
    popper_y_seed = np.vstack(observations_list)[:,numIVs:(numIVs+numDVs)]

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
                "model": model_seed,
                "x_train": popper_x_seed,
                "y_train": popper_y_seed,
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
    popper_cycle = Cycle(
        metadata=study_metadata,
        theorist=bms_theorist,
        experimentalist=popper_experimentalist,
        experiment_runner=tinkerforge_experiment_runner,
    )

    # popper_cycle.data.observations.append(np.column_stack([x_seed, y_seed]))
    # popper_cycle.data.conditions.append(x_seed)
    # popper_cycle.data.theories.append(theory_seed)

    for obs in observations_list:
        popper_cycle.data.observations.append(obs)
    for cond in conditions_list:
        popper_cycle.data.conditions.append(cond)
    for theorist in theorist_list:
        popper_cycle.data.theories.append(theorist)

    popper_cycle.run(num_cycles=1)

    model_seed = popper_cycle.data.theories[-1].model_
    theory_seed = popper_cycle.data.theories[-1]
    # append x_seed with new x
    x_seed = popper_cycle.data.observations[-1][-samples_per_cycle:, :numIVs]
    y_seed = popper_cycle.data.observations[-1][-samples_per_cycle:, numIVs:(numIVs+numDVs)]

    # x_seed = np.vstack([x_seed, popper_cycle.data.observations[-1][-samples_per_cycle:, :numIVs]])
    # # append y_seed with new y
    # if y_seed.ndim == 1:
    #     y_seed = y_seed.reshape(-1, 1)
    # y_seed = np.vstack([y_seed, popper_cycle.data.observations[-1][-samples_per_cycle:, numIVs:(numIVs+numDVs)]])


str(popper_cycle.data.theories[-1].model_), \
popper_cycle.data.theories[-1].model_.fit_par[str(popper_cycle.data.theories[-1].model_)]

all_obs = np.row_stack(popper_cycle.data.observations)
x_obs, y_obs = all_obs[:, 0], all_obs[:, 1]
plt.scatter(x_obs, y_obs, s=10, label="collected data")

# plot output of architecture search
all_obs = np.row_stack(seed_cycle.data.observations)
x_obs, y_obs = all_obs[:, 0], all_obs[:, 1]
plt.scatter(x_obs, y_obs, s=10, label="seed data")

x_pred = np.array(study_metadata.independent_variables[0].allowed_values).reshape(
    ground_truth_resolution, 1
)
y_pred_seed = seed_cycle.data.theories[0].predict(x_pred)
y_pred_final = popper_cycle.data.theories[0].predict(x_pred)
plt.plot(x_pred, y_pred_seed, color="blue", label="seed model")
plt.plot(x_pred, y_pred_final, color="red", label="final model")
plt.xlabel("Source Voltage (V)")
plt.ylabel("Current (mA)")
plt.legend()
plt.show()

def plot_status(cycle_data,
                meta_data,
                xlabel="x",
                ylabel="y",
                cycle: int=1,
                plot_contents: List[str] = ["observations_old", "theory", "condition"],
                save=False):

    axColors = dict()
    axColors["observations_old"] = "#000000" #7F7F7F
    axColors["observations_new"] = "#9FC487" #B0E997" #748F62"
    axColors["model"] = "#8AC0E9" #5A728A"
    linewidth = 5
    linewidth_conditions = 3
    markersize_old = 40
    markersize_new = 80
    fontsize = 16
    fontsize_gca = 14

    numIVs = len(meta_data.independent_variables)
    numDVs = len(meta_data.dependent_variables)


    if "observations_old" in plot_contents:
        # plot observations
        all_obs = np.row_stack(cycle_data.observations[:(cycle+1)])
        x_obs, y_obs = all_obs[:, 0:(numIVs)], all_obs[:, numIVs:(numIVs+numDVs)]
        plt.scatter(x_obs, y_obs, s=markersize_old, color=axColors["observations_old"], label="Prior Observations")

    latex_eqn = ""
    if "theory" in plot_contents:
        # plot model
        x_pred = np.array(meta_data.independent_variables[0].allowed_values).reshape(
            ground_truth_resolution, 1
        )
        y_pred = cycle_data.theories[cycle].predict(x_pred)
        latex_eqn = '$' + cycle_data.theories[cycle].model_.latex() + '$'
        plt.plot(x_pred, y_pred, linewidth=linewidth, color=axColors["model"], label="Current Model")

    if "condition" in plot_contents:
        if cycle+1 >= len(cycle_data.observations):
            raise ValueError("Cycle number is too high to plot the requested conditions")
        # plot condition
        # x_cond = cycle_data.conditions[cycle+1]
        x_cond = cycle_data.observations[cycle+1][-samples_per_cycle:, 0:(numIVs)]
        # plot vertical line for each condition in x_cond
        for idx, x in enumerate(x_cond):
            if idx == 0:
                plt.axvline(x=x, linewidth=linewidth_conditions, color=axColors["observations_new"], label="New Experiment Conditions")
            else:
                plt.axvline(x=x, linewidth=linewidth_conditions, color=axColors["observations_new"])

    if "observations_new" in plot_contents:
        # plot observations
        all_obs = cycle_data.observations[cycle+1]
        x_obs, y_obs = all_obs[:, 0:(numIVs)], all_obs[:, numIVs:(numIVs+numDVs)]
        plt.scatter(x_obs, y_obs, marker="*", s=markersize_new, label="New Observations", color=axColors["observations_new"])

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if latex_eqn != "":
        plt.title("Cycle " + str(cycle) + ", Current Model: " + latex_eqn, fontsize=fontsize)
    else:
        plt.title("Cycle " + str(cycle), fontsize=fontsize)
    plt.legend(fontsize=fontsize_gca)
    plt.xticks(fontsize=fontsize_gca)
    plt.yticks(fontsize=fontsize_gca)

    if save:
        # concatenate strings in list with underscore
        plot_contents_str = "_".join(plot_contents)
        plt.savefig("cycle_" + str(cycle) + "_" + plot_contents_str + ".png")

    plt.show()


# generate individual plots for each cycle
for cycle in range(num_popper_cycles):
    plot_status(popper_cycle.data,
                study_metadata,
                xlabel="Source Voltage (V)",
                ylabel="Current (mA)",
                cycle=cycle,
                plot_contents=["observations_old"],
                save=True)

    plot_status(popper_cycle.data,
                study_metadata,
                xlabel="Source Voltage (V)",
                ylabel="Current (mA)",
                cycle=cycle,
                plot_contents=["observations_old", "theory"],
                save=True)

    if cycle >= (num_popper_cycles):
        break

    plot_status(popper_cycle.data,
                study_metadata,
                xlabel="Source Voltage (V)",
                ylabel="Current (mA)",
                cycle=cycle,
                plot_contents=["observations_old", "theory", "condition"],
                save=True)

    plot_status(popper_cycle.data,
                study_metadata,
                xlabel="Source Voltage (V)",
                ylabel="Current (mA)",
                cycle=cycle,
                plot_contents=["observations_old", "theory", "observations_new"],
                save=True)



# make fully animated plot

x = list()
y = list()
models = list()

for cycle in range(num_popper_cycles):
     # get all observations up to and including the current cycle
     all_obs = popper_cycle.data.observations[cycle]
     x_obs, y_obs = all_obs[-samples_per_cycle:, 0:(numIVs)], all_obs[-samples_per_cycle:, numIVs:(numIVs + numDVs)]
     # reshape x_obs to be a 1D array
     x_obs = x_obs.reshape(samples_per_cycle)
     y_obs = y_obs.reshape(samples_per_cycle)
     x.append(x_obs)
     y.append(y_obs)
     models.append(popper_cycle.data.theories[cycle])



# def generate_data(setSize = 10, n_cycles = 40):
#     x = [np.random.rand(setSize)*10 for cycle in range(n_cycles)]
#     y = [[xi*2+(np.random.rand(1)*10) for xi in x[xIndex]] for xIndex in range(len(x))]
#
#     return x, y

fig = plt.figure(figsize=(7, 5))
ax = plt.subplot(111)
axColors = ["#000000", "#9FC487", "#8AC0E9"]
pltColors = [axColors[1]] * len(x[0])
currentX = x[0]
currentY = y[0]
predX = allowed_values.reshape(-1, 1)

args = dict(
    x = x,
    y = y,
    currentX = currentX,
    currentY = currentY,
    modelX = predX,
    models = models,
    axColors = axColors,
    pltColors = pltColors
    )

def update(dataIndex, ax, args):
    # Unpack argument
    linewidth = 5
    markersize_old = 40
    markersize_new = 80
    fontsize = 16
    fontsize_gca = 14

    # Unpack argument
    x = args['x']
    y = args['y']
    currentX = args['currentX']
    currentY = args['currentY']
    pltColors = args['pltColors']
    models = args['models']
    modelX = args['modelX']

    # Generate and update
    currentX = np.concatenate((currentX, x[dataIndex]))
    currentY = np.concatenate((currentY, y[dataIndex]))
    pltColors = [axColors[0]] * len(pltColors)
    pltColors = np.concatenate((pltColors, [axColors[1]] * len(x[0])))
    model = models[dataIndex]

    # Clear plot and adjust characteristics
    plt.cla()

    # Update model
    x_pred = modelX
    y_pred = model.predict(x_pred)

    # Plot animation frame
    ax.scatter(currentX, currentY, c=pltColors)
    ax.scatter(999, 999, c=axColors[1])  # This is a dummy scatter, only needed for the legend
    if dataIndex < (num_popper_cycles-1):
        ax.plot(x_pred, y_pred, c=axColors[2], alpha=.4, label='Current Model',
                linewidth=linewidth)
    else:
        ax.plot(x_pred, y_pred, c=axColors[2], label='Current Model',
                linewidth=linewidth)
    latex_eqn = '$' + model.model_.latex() + '$'

    # Format legend
    plt.legend(['Current Model', 'New Observations', 'Prior Observations'], loc='upper left', fontsize=fontsize_gca)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(axColors[2])
    leg.legendHandles[1].set_color(axColors[1])
    leg.legendHandles[2].set_color(axColors[0])

    # Format plot
    # plt.title('Cycle ' + str(dataIndex + 1) + ', Current Model: ' + latex_eqn, fontsize=fontsize)
    plt.annotate('Cycle ' + str(dataIndex + 1) + ', Current Model: ' + latex_eqn,
                 xy=(9, 0),
                 fontsize=fontsize)
    plt.xlabel("Source Voltage (V)", fontsize=fontsize)
    plt.ylabel("Current (mA)", fontsize=fontsize)
    plt.xlim([0, 3.5])
    plt.ylim([0, 0.5])
    plt.xticks(fontsize=fontsize_gca)
    plt.yticks(fontsize=fontsize_gca)


    # Re-create args
    args['currentX'] = currentX
    args['currentY'] = currentY
    args['pltColors'] = pltColors

    return ax, args

# Run animation
anim = FuncAnimation(fig, partial(update, ax=ax, args=args), frames=np.arange(0, len(args['x'])),
                     interval=1000)
plt.show()

# Save animation as gif
f = r"closedLoopAnimation.gif"
writergif = animation.PillowWriter(fps=2)
anim.save(f, writer=writergif)
