import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# exponential learning curve
exp_learning_resolution = 100
exp_learning_minimum_trial = 1
exp_learning_maximum_trial = exp_learning_resolution
exp_learning_minimum_initial_value = 0
exp_learning_maximum_initial_value = 0.5
exp_learning_lr = 0.03
exp_learning_p_asymptotic = 1.0

# Exponential learning curve

def exp_learning_metadata():
    p_initial = IV(
        name="P_asymptotic",
        allowed_values=np.linspace(exp_learning_minimum_initial_value,
                                   exp_learning_maximum_initial_value,
                                   exp_learning_resolution),
        value_range=(exp_learning_minimum_initial_value,
                     exp_learning_maximum_initial_value),
        units="performance",
        variable_label="Asymptotic Performance",
        type=ValueType.REAL
    )

    trial = IV(
        name="trial",
        allowed_values=np.linspace(exp_learning_minimum_trial,
                                   exp_learning_maximum_trial,
                                   exp_learning_resolution),
        value_range=(exp_learning_minimum_trial,
                     exp_learning_maximum_trial),
        units="trials",
        variable_label="Trials",
        type=ValueType.REAL
    )

    performance = DV(
        name="performance",
        value_range=(0, exp_learning_p_asymptotic),
        units="performance",
        variable_label="Performance",
        type=ValueType.REAL
    )

    metadata = VariableCollection(
        independent_variables=[p_initial, trial],
        dependent_variables=[performance],
    )

    return metadata

def exp_learning_experiment(X: np.ndarray,
                             p_asymptotic: float = exp_learning_p_asymptotic,
                             lr: float = exp_learning_lr,
                             std = added_noise):
    Y = np.zeros((X.shape[0],1))

    # exp learning function according to
    # Heathcote, A., Brown, S., & Mewhort, D. J. (2000). The power law repealed:
    # The case for an exponential law of practice. Psychonomic bulletin & review, 7(2), 185–207.

    # Thurstone, L. L. (1919). The learning curve equation. Psy- chological Monographs, 26(3), i.

    for idx, x in enumerate(X):
        p_initial = x[0]
        trial = x[1]
        y = p_asymptotic - (p_asymptotic - p_initial) * np.exp(- lr * trial) + np.random.normal(0, std)
        Y[idx] = y

    return Y

def exp_learning_data(metadata):

    p_initial_values = metadata.independent_variables[0].allowed_values
    trial_values = metadata.independent_variables[1].allowed_values

    X = np.array(np.meshgrid(p_initial_values, trial_values)).T.reshape(-1,2)
    y = exp_learning_experiment(X, std=0)

    return X, y

def plot_exp_learning(model = None):
    import matplotlib.pyplot as plt
    metadata = exp_learning_metadata()

    P_0_list = [0, 0.25, 0.5]
    trial = metadata.independent_variables[1].allowed_values

    for P_0 in P_0_list:
        X = np.zeros((len(trial), 2))
        X[:, 0] = P_0
        X[:, 1] = trial

        y = exp_learning_experiment(X, std=0)
        plt.plot(trial, y, label=f"$P_0 = {P_0}$ (Original)")
        if model is not None:
            y = model.predict(X)
            plt.plot(trial, y, label=f"$P_0 = {P_0}$ (Recovered)", linestyle="--")

    x_limit = [0, metadata.independent_variables[1].value_range[1]]
    y_limit = [0, 1]
    x_label = "Trial $t$"
    y_label = "Performance $P_n$"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=4, fontsize="medium")
    plt.title("Exponential Learning", fontsize="x-large")
    plt.show()

# X, y = exp_learning_data(exp_learning_metadata())
