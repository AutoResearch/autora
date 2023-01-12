import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# EVC COGED parameters
evc_coged_resolution = 20
evc_cost_parameter = 2
evc_reward_sensitivity = 0.1
evc_minimum_automaticity = 0.0
evc_maximum_automaticity = 2.0
evc_minimum_baseline_reward = 1
evc_maximum_baseline_reward = 5
evc_maximum_control_signal = 10
evc_minimum_control_signal = 0
evc_control_sginal_resolution = 1000
evc_reward_resolution = 1000

# EVC-COGED Simulation following
# Musslick, Cohen, Shenhav (2019). Decomposing Individual Differences in Cognitive Control:
# A Model-Based Approach

def evc_coged_metadata():
    task_automaticity_high_demand = IV(
        name="task_automaticity_high_demand",
        allowed_values=np.linspace(evc_minimum_automaticity, evc_maximum_automaticity,
                                   evc_coged_resolution),
        value_range=(evc_minimum_automaticity, evc_maximum_automaticity),
        units="task strength",
        variable_label="Automaticity of High-Demanding Task",
        type=ValueType.REAL
    )

    task_automaticity_low_demand = IV(
        name="task_automaticity_low_demand",
        allowed_values=np.linspace(evc_minimum_automaticity, evc_maximum_automaticity,
                                   evc_coged_resolution),
        value_range=(evc_minimum_automaticity, evc_maximum_automaticity),
        units="task strength",
        variable_label="Automaticity of Low-Demanding Task",
        type=ValueType.REAL
    )

    baseline_reward_high_demand = IV(
        name="baseline_reward_high_demand",
        allowed_values=np.linspace(evc_minimum_baseline_reward, evc_maximum_baseline_reward,
                                   evc_coged_resolution),
        value_range=(evc_minimum_baseline_reward, evc_maximum_baseline_reward),
        units="dollar",
        variable_label="Reward of High-Demanding Task",
        type=ValueType.REAL
    )

    indifference_value = DV(
        name="indifference_value",
        value_range=(0, evc_maximum_baseline_reward),
        units="dollar",
        variable_label="Indifference Value",
        type=ValueType.REAL
    )

    metadata = VariableCollection(
        independent_variables=[task_automaticity_high_demand,
                               task_automaticity_low_demand,
                               baseline_reward_high_demand],
        dependent_variables=[indifference_value],
    )

    return metadata

def evc_coged_experiment(X: np.ndarray,
                                 cost_parameter: float = evc_cost_parameter,
                                 reward_sensitivity: float = evc_reward_sensitivity,
                                 std = added_noise):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        automaticity_high_demand = x[0]
        automaticity_low_demand = x[1]
        baseline_reward_high_demand = x[2]

        # evc of high-demanding task
        (evc_high_demand, opt_signal) = compute_evc(baseline_reward_high_demand,
                                      automaticity_high_demand,
                                      cost_parameter,
                                      reward_sensitivity)
        evc_high_demand += np.random.normal(0, std)

        # find a reward condition for which the EVC agent is indifferent between choosing the low-
        # versus high-demanding task
        rewards = np.linspace(0, baseline_reward_high_demand, evc_reward_resolution)
        indifference_value = baseline_reward_high_demand
        for reward in rewards:
            (evc_low_demand, opt_signal) = compute_evc(reward,
                                         automaticity_low_demand,
                                         cost_parameter,
                                         reward_sensitivity)
            evc_low_demand += np.random.normal(0, std)
            if evc_low_demand >= evc_high_demand:
                indifference_value = reward
                break

        Y[idx] = indifference_value

    return Y

def compute_evc(reward: float,
                task_automaticity: float,
                cost_parameter: float = evc_cost_parameter,
                reward_sensitivity: float = evc_reward_sensitivity):

    signals = np.linspace(evc_minimum_control_signal,
                          evc_maximum_control_signal,
                          evc_control_sginal_resolution)

    costs = np.exp(cost_parameter * signals)
    rewards = reward_sensitivity * reward
    performance = 1 / (1 + np.exp(-(signals + task_automaticity)))

    evc = rewards * performance - costs

    # identify evc for optimal signal
    max_evc = np.max(evc)
    # import matplotlib.pyplot as plt
    # plt.plot(signals, evc)
    # plt.show()

    opt_signal = signals[np.where(evc == max_evc)]
    if max_evc == evc[-1]:
        print("Warning: Selected maximum control signal intensity.")

    return (max_evc, opt_signal)

def evc_coged_data(metadata):

    task_automaticity_high_demand = metadata.independent_variables[0].allowed_values
    task_automaticity_low_demand = metadata.independent_variables[1].allowed_values
    baseline_reward_high_demand = metadata.independent_variables[2].allowed_values

    X = np.array(np.meshgrid(task_automaticity_high_demand,
                             task_automaticity_low_demand,
                             baseline_reward_high_demand)).T.reshape(-1,3)

    # remove all lines in X if high-demanding task is more automatic than low-demanding task
    X = X[X[:,0] < X[:,1]]


    y = evc_coged_experiment(X, std=0)

    return X, y

def plot_evc_coged(model = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    task_automaticity_low_demand_list = [0.5, 1, 2]
    baseline_reward_high_demand = 1.0
    task_automaticity_high_demand = np.linspace(0,
                                                0.5, 100)
    colors = mcolors.TABLEAU_COLORS
    col_keys = list(colors.keys())
    for idx, task_automaticity_low_demand in enumerate(task_automaticity_low_demand_list):

        X = np.zeros((len(task_automaticity_high_demand), 3))
        X[:, 0] = task_automaticity_high_demand
        X[:, 1] = task_automaticity_low_demand
        X[:, 2] = baseline_reward_high_demand

        task_difficulty = 1-task_automaticity_high_demand
        y = evc_coged_experiment(X, std=0)
        subjective_value = y / baseline_reward_high_demand

        plt.plot(task_difficulty,
                 subjective_value,
                 label=f"Automaticity of Easy Task = {task_automaticity_low_demand} (Original)",
                 c=colors[col_keys[idx]])
        if model is not None:
            y = model.predict(X)
            subjective_value = y / baseline_reward_high_demand
            plt.plot(task_difficulty,
                     subjective_value,
                     label=f"Automaticity of Easy Task = {task_automaticity_low_demand} (Recovered)",
                     c=colors[col_keys[idx]], linestyle="--")

    x_limit = [np.min(task_difficulty), np.max(task_difficulty)]
    y_limit = [0, 1]
    x_label = "Difficulty of Task"
    y_label = "Subjective Value"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=3, fontsize="medium")
    plt.title("Cognitive Effort Discounting", fontsize="x-large")
    plt.show()


# X, y = evc_coged_data(evc_coged_metadata())
# plot_evc_coged()
