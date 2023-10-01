import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# stevens' power law parameters
stevens_resolution = 100
stevens_proportionality_constant = 1.0
stevens_modality_constant = 0.8
maximum_stimulus_intensity = 5.0

# Stevens' Power Law

def stevens_power_law_metadata():
    iv1 = IV(
        name="S",
        allowed_values=np.linspace(1/stevens_resolution, maximum_stimulus_intensity, stevens_resolution),
        value_range=(1/stevens_resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus Intensity",
        type=ValueType.REAL
    )

    dv1 = DV(
        name="perceived_intensity",
        value_range=(0, maximum_stimulus_intensity),
        units="sensation",
        variable_label="Perceived Intensity",
        type=ValueType.REAL
    )

    metadata = VariableCollection(
        independent_variables=[iv1],
        dependent_variables=[dv1],
    )

    return metadata

def stevens_power_law_experiment(X: np.ndarray,
                             stevens_proportionality_constant: float = stevens_proportionality_constant,
                             stevens_modality_constant: float = stevens_modality_constant,
                             std = added_noise):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):
        y = stevens_proportionality_constant * x[0]**stevens_modality_constant + np.random.normal(0, std)
        Y[idx] = y

    return Y

def stevens_power_law_data(metadata, std=added_noise):

    s_values = metadata.independent_variables[0].allowed_values

    X = np.array(np.meshgrid(s_values)).T.reshape(-1,1)

    y = stevens_power_law_experiment(X, std=std)

    return X, y

def plot_stevens_power_law(model = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = mcolors.TABLEAU_COLORS
    col_keys = list(colors.keys())
    metadata = stevens_power_law_metadata()
    X, y = stevens_power_law_data(metadata)
    plt.plot(X, y, label="Original", c=colors[col_keys[0]])
    if model is not None:
        y = model.predict(X)
        plt.plot(X, y, label=f"Recovered", c=colors[col_keys[0]], linestyle="--")
    x_limit = [0, metadata.independent_variables[0].value_range[1]]
    y_limit = [0, 4]
    x_label = "Stimulus Intensity"
    y_label = "Perceived Stimulus Intensity"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Stevens' Power Law", fontsize="x-large")
    plt.show()


# X, y = stevens_power_law_data(stevens_power_law_metadata())
