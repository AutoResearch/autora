import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0

# EVC COGED parameters
evc_congruency_resolution = 10
evc_cost_parameter = 0.05
evc_reward_sensitivity = 0.1
evc_minimum_coherence = -1.0
evc_maximum_coherence = 1.0
evc_coherence_scalar = 4
evc_minimum_reward = 1
evc_maximum_reward = 3
evc_maximum_control_signal = 1
evc_minimum_control_signal = 0
evc_control_sginal_resolution = 1000

# EVC-Congruency Simulation following
# Musslick, Cohen, Shenhav (2019). Decomposing Individual Differences in Cognitive Control:
# A Model-Based Approach

def evc_congruency_metadata():
    color_coherence = IV(
        name="color_coherence",
        allowed_values=np.linspace(evc_minimum_coherence, evc_maximum_coherence,
                                   evc_congruency_resolution),
        value_range=(evc_minimum_coherence, evc_maximum_coherence),
        units="coherence",
        variable_label="Coherence of Color",
        type=ValueType.REAL
    )

    motion_coherence = IV(
        name="motion_coherence",
        allowed_values=np.linspace(evc_minimum_coherence, evc_maximum_coherence,
                                   evc_congruency_resolution),
        value_range=(evc_minimum_coherence, evc_maximum_coherence),
        units="coherence",
        variable_label="Coherence of Motion",
        type=ValueType.REAL
    )

    color_reward = IV(
        name="baseline_reward_high_demand",
        allowed_values=np.linspace(evc_minimum_reward, evc_maximum_reward,
                                   evc_congruency_resolution),
        value_range=(evc_minimum_reward, evc_maximum_reward),
        units="dollar",
        variable_label="Reward of High-Demanding Task",
        type=ValueType.REAL
    )

    choose_left = DV(
        name="choose_left",
        value_range=(0, 1),
        units="probability",
        variable_label="Choose Left",
        type=ValueType.PROBABILITY
    )

    metadata = VariableCollection(
        independent_variables=[color_coherence, motion_coherence, color_reward],
        dependent_variables=[choose_left],
    )

    return metadata

def evc_congruency_experiment(X: np.ndarray,
                                 cost_parameter: float = evc_cost_parameter,
                                 reward_sensitivity: float = evc_reward_sensitivity,
                                 std = added_noise):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        color_coherence = x[0]
        motion_coherence = x[1]
        color_reward = x[2]

        # evc of color task
        (evc_color, opt_signal, opt_choice) = compute_evc(color_reward,
                                                  color_coherence,
                                                  motion_coherence,
                                                  cost_parameter,
                                                  reward_sensitivity,
                                                  std)

        Y[idx] = opt_choice

    return Y

def compute_evc(reward: float,
                task_coherence: float,
                distractor_coherence: float,
                cost_parameter: float = evc_cost_parameter,
                reward_sensitivity: float = evc_reward_sensitivity,
                std = added_noise):

    signals = np.linspace(evc_minimum_control_signal,
                          evc_maximum_control_signal,
                          evc_control_sginal_resolution)

    costs = np.exp(cost_parameter * signals)
    rewards = reward_sensitivity * reward
    input = evc_coherence_scalar * task_coherence * signals + \
            evc_coherence_scalar * distractor_coherence * (1-signals) + \
            np.random.normal(0, std)
    choice = 1 / (1 + np.exp(-input))
    performance = np.zeros(choice.shape)
    performance[:] = choice
    performance[task_coherence < 0] = 1 - choice[task_coherence < 0]

    evc = rewards * performance - costs

    # identify evc for optimal signal
    max_evc = np.max(evc)
    # import matplotlib.pyplot as plt
    # plt.plot(signals, performance, c='r')
    # plt.show()
    # plt.plot(signals, evc)
    # plt.show()

    opt_signal = signals[np.where(evc == max_evc)]
    opt_choice = choice[np.where(evc == max_evc)][0]
    if max_evc == evc[-1]:
        print("Warning: Selected maximum control signal intensity.")

    return (max_evc, opt_signal, opt_choice)

def evc_congruency_data(metadata):

    color_coherence = metadata.independent_variables[0].allowed_values
    motion_coherence = metadata.independent_variables[1].allowed_values
    color_reward = metadata.independent_variables[2].allowed_values

    X = np.array(np.meshgrid(color_coherence,
                             motion_coherence,
                             color_reward)).T.reshape(-1,3)

    y = evc_congruency_experiment(X, std=0)

    return X, y


def plot_evc_congruency(model = None):
    import matplotlib.pyplot as plt

    metadata = evc_congruency_metadata()

    color_rewards = [1, 1.5, 3]
    color_coherence = np.linspace(metadata.independent_variables[0].value_range[0],
                                  metadata.independent_variables[0].value_range[1], 100)
    motion_coherence = -0.2

    for color_reward in color_rewards:

        X = np.zeros((len(color_coherence), 3))
        X[:, 0] = color_coherence
        X[:, 1] = motion_coherence
        X[:, 2] = color_reward

        y = evc_congruency_experiment(X, std=0)
        plt.plot(color_coherence, y,
                 label=f"Reward = {color_reward} (Original)")
        if model is not None:
            y = model.predict(X)
            plt.plot(color_coherence, y,
                     label=f"Reward = {color_reward} (Recovered)")

    x_limit = [np.min(color_coherence), np.max(color_coherence)]
    y_limit = [0, 1]
    x_label = f"Color Coherence (Motion Coherence = {motion_coherence})"
    y_label = "Probability of Left Response"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Psychometric Curve", fontsize="x-large")
    plt.show()


# X, y = evc_congruency_data(evc_congruency_metadata())
# plot_evc_congruency()
