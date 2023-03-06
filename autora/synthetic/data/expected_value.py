from functools import partial

import numpy as np

from autora.variable import DV, IV, ValueType, VariableCollection

from .._inventory import SyntheticExperimentCollection, register


def get_metadata(minimum_value, maximum_value, resolution):
    v_a = IV(
        name="V_A",
        allowed_values=np.linspace(
            minimum_value,
            maximum_value,
            resolution,
        ),
        value_range=(minimum_value, maximum_value),
        units="dollar",
        variable_label="Value of Option A",
        type=ValueType.REAL,
    )

    v_b = IV(
        name="V_B",
        allowed_values=np.linspace(
            minimum_value,
            maximum_value,
            resolution,
        ),
        value_range=(minimum_value, maximum_value),
        units="dollar",
        variable_label="Value of Option B",
        type=ValueType.REAL,
    )

    p_a = IV(
        name="P_A",
        allowed_values=np.linspace(0, 1, resolution),
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Option A",
        type=ValueType.REAL,
    )

    p_b = IV(
        name="P_B",
        allowed_values=np.linspace(0, 1, resolution),
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

    metadata_ = VariableCollection(
        independent_variables=[v_a, p_a, v_b, p_b],
        dependent_variables=[dv1],
    )
    return metadata_


def expected_value_theory(
    name="Expected Value Theory",
    choice_temperature: float = 0.1,
    value_lambda: float = 0.5,
    resolution=10,
    minimum_value=-1,
    maximum_value=1,
    added_noise: float = 0.01,
    rng=np.random.default_rng(),
):

    params = dict(
        name=name,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        resolution=resolution,
        choice_temperature=choice_temperature,
        value_lambda=value_lambda,
        added_noise=added_noise,
        random_number_generator=rng,
    )

    metadata = get_metadata(
        minimum_value=minimum_value, maximum_value=maximum_value, resolution=resolution
    )

    def experiment_runner(X: np.ndarray, added_noise_=added_noise):

        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            value_A = value_lambda * x[0]
            value_B = value_lambda * x[2]

            probability_a = x[1]
            probability_b = x[3]

            expected_value_A = value_A * probability_a + rng.normal(0, added_noise_)
            expected_value_B = value_B * probability_b + rng.normal(0, added_noise_)

            # compute probability of choosing option A
            p_choose_A = np.exp(expected_value_A / choice_temperature) / (
                np.exp(expected_value_A / choice_temperature)
                + np.exp(expected_value_B / choice_temperature)
            )

            Y[idx] = p_choose_A

        return Y

    ground_truth = partial(experiment_runner, added_noise_=0.0)

    def domain():
        X = np.array(
            np.meshgrid([x.allowed_values for x in metadata.independent_variables])
        ).T.reshape(-1, 4)
        return X

    def plotter(model=None):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

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

            y = ground_truth(X)
            colors = mcolors.TABLEAU_COLORS
            col_keys = list(colors.keys())
            plt.plot(
                p_a, y, label=f"$V(A) = {v_a}$ (Original)", c=colors[col_keys[idx]]
            )
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
        plt.title(name, fontsize="x-large")
        plt.show()

    collection = SyntheticExperimentCollection(
        name=name,
        metadata=metadata,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
    )
    return collection


register(id="expected_value", closure=expected_value_theory)
