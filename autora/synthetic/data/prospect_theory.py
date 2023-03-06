from functools import partial

import numpy as np

from .._inventory import SyntheticExperimentCollection, register
from .expected_value import get_metadata


def prospect_theory(
    added_noise=0.00,
    choice_temperature=0.1,
    value_alpha=0.88,
    value_beta=0.88,
    value_lambda=2.25,
    probability_alpha=1.0,
    probability_beta=0.69,
    resolution=10,
    minimum_value=-1,
    maximum_value=1,
    rng=np.random.default_rng(),
    name="Prospect Theory",
):
    """
    Parameters from
    D. Kahneman, A. Tversky, Prospect theory: An analysis of decision under risk.
    Econometrica 47, 263–292 (1979). doi:10.2307/1914185

    Power value function according to:

    A. Tversky, D. Kahneman, Advances in prospect theory: Cumulative representation of
    uncertainty. J. Risk Uncertain. 5, 297–323 (1992). doi:10.1007/BF00122574

    I. Gilboa, Expected utility with purely subjective non-additive probabilities.
    J. Math. Econ. 16, 65–88 (1987). doi:10.1016/0304-4068(87)90022-X

    D. Schmeidler, Subjective probability and expected utility without additivity.
    Econometrica 57, 571 (1989). doi:10.2307/1911053

    """

    params = dict(
        added_noise=added_noise,
        choice_temperature=choice_temperature,
        value_alpha=value_alpha,
        value_beta=value_beta,
        value_lambda=value_lambda,
        probability_alpha=probability_alpha,
        probability_beta=probability_beta,
        resolution=resolution,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        rng=rng,
        name=name,
    )

    metadata = get_metadata(
        minimum_value=minimum_value, maximum_value=maximum_value, resolution=resolution
    )

    def experiment(
        X: np.ndarray,
        std=added_noise,
        rng_generator=np.random.default_rng(),
    ):

        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):

            # compute value of option A
            if x[0] > 0:
                value_A = x[0] ** value_alpha
            else:
                value_A = -value_lambda * (-x[0]) ** (value_beta)

            # compute value of option B
            if x[2] > 0:
                value_B = x[2] ** value_alpha
            else:
                value_B = -value_lambda * (-x[2]) ** (value_beta)

            # probability function according to:

            # A. Tversky, D. Kahneman, Advances in prospect theory: Cumulative representation of
            # uncertainty. J. Risk Uncertain. 5, 297–323 (1992). doi:10.1007/BF00122574

            # compute probability of option A
            if x[1] >= 0:
                coefficient = probability_alpha
            else:
                coefficient = probability_beta

            probability_a = x[1] ** coefficient / (
                x[1] ** coefficient + (1 - x[1]) ** coefficient
            ) ** (1 / coefficient)

            # compute probability of option B
            if x[3] >= 0:
                coefficient = probability_alpha
            else:
                coefficient = probability_beta

            probability_b = x[3] ** coefficient / (
                x[3] ** coefficient + (1 - x[3]) ** coefficient
            ) ** (1 / coefficient)

            expected_value_A = value_A * probability_a + rng_generator.normal(0, std)
            expected_value_B = value_B * probability_b + rng_generator.normal(0, std)

            # compute probability of choosing option A
            p_choose_A = np.exp(expected_value_A / choice_temperature) / (
                np.exp(expected_value_A / choice_temperature)
                + np.exp(expected_value_B / choice_temperature)
            )

            Y[idx] = p_choose_A

        return Y

    ground_truth = partial(experiment, std=0.0)

    domain = np.array(
        np.meshgrid([x.allowed_values for x in metadata.independent_variables])
    ).T.reshape(-1, 4)

    def plotter(model=None):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        v_a_list = [-0.5, 0.5, 1]
        p_a = np.linspace(0, 1, 100)

        v_b = 0.5
        p_b = 0.5

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
        params=params,
        metadata=metadata,
        domain=domain,
        experiment=experiment,
        ground_truth=ground_truth,
        plotter=plotter,
    )
    return collection


register("prospect_theory", prospect_theory)
