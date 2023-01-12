import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# weber-fechner parameters
weber_resolution = 100
weber_constant = 1.0
maximum_stimulus_intensity = 5.0

# Weber-Fechner-Law

def weber_fechner_metadata():
    iv1 = IV(
        name="S1",
        allowed_values=np.linspace(1/weber_resolution, maximum_stimulus_intensity, weber_resolution),
        value_range=(1/weber_resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus 1 Intensity",
        type=ValueType.REAL
    )

    iv2 = IV(
        name="S2",
        allowed_values=np.linspace(1/weber_resolution, maximum_stimulus_intensity, weber_resolution),
        value_range=(1/weber_resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus 2 Intensity",
        type=ValueType.REAL
    )

    dv1 = DV(
        name="difference_detected",
        value_range=(0, maximum_stimulus_intensity),
        units="sensation",
        variable_label="Sensation",
        type=ValueType.REAL
    )

    metadata = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    return metadata

def weber_fechner_experiment(X: np.ndarray,
                             weber_constant: float = weber_constant,
                             std = added_noise):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):
        # jnd =  np.min(x) * weber_constant
        # response = (x[1]-x[0]) - jnd
        # y = 1/(1+np.exp(-response)) + np.random.normal(0, std)
        y = weber_constant * np.log(x[1]/x[0]) + np.random.normal(0, std)
        Y[idx] = y

    return Y

def weber_fechner_data(metadata):

    s1_values = metadata.independent_variables[0].allowed_values
    s2_values = metadata.independent_variables[1].allowed_values

    X = np.array(np.meshgrid(s1_values, s2_values)).T.reshape(-1,2)
    # remove all combinations where s1 > s2
    X = X[X[:,0] <= X[:,1]]

    y = weber_fechner_experiment(X, std=0)

    return X, y

def plot_weber_fechner(model = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = mcolors.TABLEAU_COLORS
    col_keys = list(colors.keys())

    metadata = weber_fechner_metadata()

    S0_list = [1, 2, 4]
    delta_S = np.linspace(0, 5, 100)

    for idx, S0_value in enumerate(S0_list):
        S0 = S0_value + np.zeros(delta_S.shape)
        S1 = S0 + delta_S
        X = np.array([S0, S1]).T
        y = weber_fechner_experiment(X, std=0)
        plt.plot(delta_S, y, label=f"$S_0 = {S0_value}$ (Original)", c=colors[col_keys[idx]])
        if model is not None:
            y = model.predict(X)
            plt.plot(delta_S, y, label=f"$S_0 = {S0_value}$ (Recovered)",
                     c=colors[col_keys[idx]], linestyle="--")

    x_limit = [0, metadata.independent_variables[0].value_range[1]]
    y_limit = [0, 2]
    x_label = "Stimulus Intensity Difference $\Delta S = S_1 - S_0$"
    y_label = "Perceived Intensity of Stimulus $S_1$"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Weber-Fechner Law", fontsize="x-large")
    plt.show()
