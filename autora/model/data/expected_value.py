import numpy as np

from autora.model.inventory import register
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# expected value theory with linear value function
expected_value_choice_temperature = 0.1
expected_value_lambda = 0.5
expected_value_resolution = 10
expected_value_minimum_value = -1
expected_value_maximum_value = 1

# basic expected value theory


def expected_value_theory_metadata():

    v_a = IV(
        name="V_A",
        allowed_values=np.linspace(
            expected_value_minimum_value,
            expected_value_maximum_value,
            expected_value_resolution,
        ),
        value_range=(expected_value_minimum_value, expected_value_maximum_value),
        units="dollar",
        variable_label="Value of Option A",
        type=ValueType.REAL,
    )

    v_b = IV(
        name="V_B",
        allowed_values=np.linspace(
            expected_value_minimum_value,
            expected_value_maximum_value,
            expected_value_resolution,
        ),
        value_range=(expected_value_minimum_value, expected_value_maximum_value),
        units="dollar",
        variable_label="Value of Option B",
        type=ValueType.REAL,
    )

    p_a = IV(
        name="P_A",
        allowed_values=np.linspace(0, 1, expected_value_resolution),
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Option A",
        type=ValueType.REAL,
    )

    p_b = IV(
        name="P_B",
        allowed_values=np.linspace(0, 1, expected_value_resolution),
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Option B",
        type=ValueType.REAL,
    )

    dv1 = DV(
        name="choose_A",
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Choosing Option A",
        type=ValueType.PROBABILITY,
    )

    metadata = VariableCollection(
        independent_variables=[v_a, p_a, v_b, p_b],
        dependent_variables=[dv1],
    )

    return metadata


def expected_value_theory_experiment(
    X: np.ndarray,
    choice_temperature: float = expected_value_choice_temperature,
    value_lambda=expected_value_lambda,
    std=added_noise,
):

    Y = np.zeros((X.shape[0], 1))
    for idx, x in enumerate(X):

        value_A = value_lambda * x[0]
        value_B = value_lambda * x[2]

        probability_a = x[1]
        probability_b = x[3]

        expected_value_A = value_A * probability_a + np.random.normal(0, std)
        expected_value_B = value_B * probability_b + np.random.normal(0, std)

        # compute probability of choosing option A
        p_choose_A = np.exp(expected_value_A / choice_temperature) / (
            np.exp(expected_value_A / choice_temperature)
            + np.exp(expected_value_B / choice_temperature)
        )

        Y[idx] = p_choose_A

    return Y


def expected_value_theory_data(metadata):

    v_a = metadata.independent_variables[0].allowed_values
    p_a = metadata.independent_variables[1].allowed_values
    v_b = metadata.independent_variables[2].allowed_values
    p_b = metadata.independent_variables[3].allowed_values

    X = np.array(np.meshgrid(v_a, p_a, v_b, p_b)).T.reshape(-1, 4)

    y = expected_value_theory_experiment(X, std=0)

    return X, y


def plot_expected_value(model=None):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    metadata = expected_value_theory_metadata()

    v_a_list = [-1, 0.5, 1]
    v_b = 0.5
    p_b = 0.5
    p_a = np.linspace(0, 1, 100)

    for idx, v_a in enumerate(v_a_list):
        X = np.zeros((len(p_a), 4))
        X[:, 0] = v_a
        X[:, 1] = p_a
        X[:, 2] = v_b
        X[:, 3] = p_b

        y = expected_value_theory_experiment(X, std=0)
        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())
        plt.plot(p_a, y, label=f"$V(A) = {v_a}$ (Original)", c=colors[col_keys[idx]])
        if model is not None:
            y = model.predict(X)
            plt.plot(
                p_a,
                y,
                label=f"$V(A) = {v_a}$ (Recovered)",
                c=colors[col_keys[idx]],
                linestyle="--",
            )

    x_limit = [0, metadata.independent_variables[1].value_range[1]]
    y_limit = [0, 1]
    x_label = "Probability of Choosing Option A"
    y_label = "Probability of Obtaining V(A)"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Expected Value Theory", fontsize="x-large")
    plt.show()


# plot_expected_value()
# X, y = expected_value_theory_data(expected_value_theory_metadata())

register(
    "expected_value",
    metadata=expected_value_theory_metadata,
    data=expected_value_theory_data,
    synthetic_experiment_runner=expected_value_theory_experiment,
    name="Expected Utility Theory",
    plotter=plot_expected_value,
)
