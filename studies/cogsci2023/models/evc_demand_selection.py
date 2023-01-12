import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# EVC COGED parameters
evc_demand_resolution = 10
evc_choice_temperature = 0.05
evc_cost_parameter = 2
evc_reward_sensitivity = 0.1
evc_minimum_automaticity = 0.0
evc_maximum_automaticity = 2.0
evc_minimum_reward = 1
evc_maximum_reward = 5
evc_maximum_control_signal = 10
evc_minimum_control_signal = 0
evc_control_sginal_resolution = 1000

# EVC-COGED Simulation following
# Musslick, Cohen, Shenhav (2019). Decomposing Individual Differences in Cognitive Control:
# A Model-Based Approach

def evc_demand_metadata():
    task_A_automaticity = IV(
        name="task_A_automaticity",
        allowed_values=np.linspace(evc_minimum_automaticity, evc_maximum_automaticity,
                                   evc_demand_resolution),
        value_range=(evc_minimum_automaticity, evc_maximum_automaticity),
        units="task strength",
        variable_label="Automaticity of Task A",
        type=ValueType.REAL
    )

    task_B_automaticity = IV(
        name="task_B_automaticity",
        allowed_values=np.linspace(evc_minimum_automaticity, evc_maximum_automaticity,
                                   evc_demand_resolution),
        value_range=(evc_minimum_automaticity, evc_maximum_automaticity),
        units="task strength",
        variable_label="Automaticity of Task B",
        type=ValueType.REAL
    )

    task_A_reward = IV(
        name="task_A_reward",
        allowed_values=np.linspace(evc_minimum_reward, evc_maximum_reward,
                                   evc_demand_resolution),
        value_range=(evc_minimum_reward, evc_maximum_reward),
        units="dollar",
        variable_label="Reward of Task A",
        type=ValueType.REAL
    )

    task_B_reward = IV(
        name="task_B_reward",
        allowed_values=np.linspace(evc_minimum_reward, evc_maximum_reward,
                                   evc_demand_resolution),
        value_range=(evc_minimum_reward, evc_maximum_reward),
        units="dollar",
        variable_label="Reward of Task B",
        type=ValueType.REAL
    )

    choose_A = DV(
        name="choose_A",
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Choosing Task A",
        type=ValueType.PROBABILITY
    )

    metadata = VariableCollection(
        independent_variables=[task_A_automaticity,
                               task_B_automaticity,
                               task_A_reward,
                               task_B_reward],
        dependent_variables=[choose_A],
    )

    return metadata

def evc_demand_experiment(X: np.ndarray,
                                 cost_parameter: float = evc_cost_parameter,
                                 reward_sensitivity: float = evc_reward_sensitivity,
                                 choice_temporature: float = evc_choice_temperature,
                                 std = added_noise):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        task_A_automaticity = x[0]
        task_B_automaticity = x[1]
        task_A_reward = x[2]
        task_B_reward = x[3]

        # evc of task A
        (evc_a, opt_signal) = compute_evc(task_A_reward,
                                      task_A_automaticity,
                                      cost_parameter,
                                      reward_sensitivity)
        evc_a += np.random.normal(0, std)

        # evc of task B
        (evc_b, opt_signal) = compute_evc(task_B_reward,
                                        task_B_automaticity,
                                        cost_parameter,
                                        reward_sensitivity)
        evc_b += np.random.normal(0, std)

        choose_A = np.exp(evc_a / choice_temporature) / \
                   (np.exp(evc_a / choice_temporature) +
                    np.exp(evc_b / choice_temporature))

        Y[idx] = choose_A

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

def evc_demand_data(metadata):

    task_A_automaticity = metadata.independent_variables[0].allowed_values
    task_B_automaticity = metadata.independent_variables[1].allowed_values
    task_A_reward = metadata.independent_variables[2].allowed_values
    task_B_reward = metadata.independent_variables[2].allowed_values

    X = np.array(np.meshgrid(task_A_automaticity,
                             task_B_automaticity,
                             task_A_reward,
                             task_B_reward)).T.reshape(-1,4)


    y = evc_demand_experiment(X, std=0)

    return X, y

def plot_evc_demand(model = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    metadata = evc_demand_metadata()

    task_A_reward_list = [1, 2, 3]
    task_B_reward = task_A_reward_list[1]
    task_A_automaticity_list = np.linspace(metadata.independent_variables[0].value_range[0],
                                           metadata.independent_variables[0].value_range[1], 1000)
    task_B_automaticity = metadata.independent_variables[0].value_range[0] \
                          + (metadata.independent_variables[0].value_range[1]
                          - metadata.independent_variables[0].value_range[0]) / 2


    for idx, task_A_reward in enumerate(task_A_reward_list):

        X = np.zeros((len(task_A_automaticity_list), 4))
        X[:, 0] = task_A_automaticity_list
        X[:, 1] = task_B_automaticity
        X[:, 2] = task_A_reward
        X[:, 3] = task_B_reward

        y = evc_demand_experiment(X, std=0)
        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())
        plt.plot(task_A_automaticity_list, y,
                 label=f"Reward for Task A = {task_A_reward}(Original)", c=colors[col_keys[idx]])
        if model is not None:
            y = model.predict(X)
            plt.plot(task_A_automaticity_list, y,
                     label=f"Reward for Task A = {task_A_reward}(Recovered)", c=colors[col_keys[idx]], linestyle="--")

    x_limit = [np.min(task_A_automaticity_list), np.max(task_A_automaticity_list)]
    y_limit = [0, 1]
    x_label = "Automaticity of Task A"
    y_label = "Probability of Choosing Task A"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Demand Avoidance", fontsize="x-large")
    plt.show()


# X, y = evc_demand_data(evc_demand_metadata())
# plot_evc_demand()
