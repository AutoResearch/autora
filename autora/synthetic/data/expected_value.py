from functools import partial

import numpy as np

from autora.variable import DV, IV, ValueType, VariableCollection

from .._inventory import SyntheticExperimentCollection, register


def make_metadata(minimum_value_, maximum_value_, resolution_):
    v_a = IV(
        name="V_A",
        allowed_values=np.linspace(
            minimum_value_,
            maximum_value_,
            resolution_,
        ),
        value_range=(minimum_value_, maximum_value_),
        units="dollar",
        variable_label="Value of Option A",
        type=ValueType.REAL,
    )

    v_b = IV(
        name="V_B",
        allowed_values=np.linspace(
            minimum_value_,
            maximum_value_,
            resolution_,
        ),
        value_range=(minimum_value_, maximum_value_),
        units="dollar",
        variable_label="Value of Option B",
        type=ValueType.REAL,
    )

    p_a = IV(
        name="P_A",
        allowed_values=np.linspace(0, 1, resolution_),
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Option A",
        type=ValueType.REAL,
    )

    p_b = IV(
        name="P_B",
        allowed_values=np.linspace(0, 1, resolution_),
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
    lambda_: float = 0.5,
    resolution=10,
    minimum_value=-1,
    maximum_value=1,
    added_noise: float = 0.01,
    rng=np.random.default_rng(),
):

    params = dict(
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        resolution=resolution,
        choice_temperature=choice_temperature,
        lambda_=lambda_,
        added_noise=added_noise,
        random_number_generator=rng,
    )

    metadata_ = make_metadata(
        minimum_value_=minimum_value,
        maximum_value_=maximum_value,
        resolution_=resolution,
    )

    def expected_value_theory_experiment(X: np.ndarray, added_noise_=added_noise):

        n_obs = X.shape[0]

        value_a = lambda_ * X[:, 0]
        value_b = lambda_ * X[:, 2]

        probability_a = X[:, 1]
        probability_b = X[:, 3]

        expected_value_a = value_a * probability_a + rng.normal(
            0, added_noise_, size=n_obs
        )
        expected_value_b = value_b * probability_b + rng.normal(
            0, added_noise_, size=n_obs
        )

        # compute probability of choosing option A
        p_choose_a = np.exp(expected_value_a / choice_temperature) / (
            np.exp(expected_value_a / choice_temperature)
            + np.exp(expected_value_b / choice_temperature)
        )

        return p_choose_a

    ground_truth_ = partial(expected_value_theory_experiment, added_noise_=0.0)

    domain = np.array(
        np.meshgrid(x.allowed_values for x in metadata_.independent_variables)
    ).T.reshape(-1, 4)

    def plotter(model=None, ground_truth=ground_truth_):
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

        x_limit = [0, metadata_.independent_variables[1].value_range[1]]
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
        metadata=metadata_,
        experiment=expected_value_theory_experiment,
        ground_truth=ground_truth_,
        domain=domain,
        plotter=plotter,
        params=params,
    )
    return collection


register(id_="expected_value", closure=expected_value_theory)
