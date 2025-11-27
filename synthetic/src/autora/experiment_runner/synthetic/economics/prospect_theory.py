from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.economics.expected_value_theory import (
    get_variables,
)
from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection


def prospect_theory(
    name="Prospect Theory",
    choice_temperature=0.1,
    value_alpha=0.88,
    value_beta=0.88,
    value_lambda=2.25,
    probability_alpha=0.61,
    probability_beta=0.69,
    resolution=10,
    minimum_value=-1,
    maximum_value=1,
):
    """
    Parameters from
    D. Kahneman, A. Tversky, Prospect theory: An analysis of decision under risk.
    Econometrica 47, 263–292 (1979). doi:10.2307/1914185

    Power value function according to:
        - A. Tversky, D. Kahneman, Advances in prospect theory: Cumulative representation of
          uncertainty. J. Risk Uncertain. 5, 297–323 (1992). doi:10.1007/BF00122574

        - I. Gilboa, Expected utility with purely subjective non-additive probabilities.
          J. Math. Econ. 16, 65–88 (1987). doi:10.1016/0304-4068(87)90022-X

        - D. Schmeidler, Subjective probability and expected utility without additivity.
          Econometrica 57, 571 (1989). doi:10.2307/1911053

    Probability function according to:
        A. Tversky, D. Kahneman, Advances in prospect theory: Cumulative representation of
        uncertainty. J. Risk Uncertain. 5, 297–323 (1992). doi:10.1007/BF00122574
    Examples:
        >>> s = prospect_theory()
        >>> s.run(np.array([[.9,.1,.1,.9]]), random_state=42)
           V_A  P_A  V_B  P_B  choose_A
        0  0.9  0.1  0.1  0.9  0.709777

    """

    params = dict(
        choice_temperature=choice_temperature,
        value_alpha=value_alpha,
        value_beta=value_beta,
        value_lambda=value_lambda,
        probability_alpha=probability_alpha,
        probability_beta=probability_beta,
        resolution=resolution,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        name=name,
    )

    variables = get_variables(
        minimum_value=minimum_value, maximum_value=maximum_value, resolution=resolution
    )

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise=0.01,
        random_state: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_state)
        X = np.array(conditions)
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            # power value function according to:

            # A. Tversky, D. Kahneman, Advances in prospect theory: Cumulative representation of
            # uncertainty. J. Risk Uncertain. 5, 297–323 (1992). doi:10.1007/BF00122574

            # I. Gilboa, Expected utility with purely subjective non-additive probabilities.
            # J. Math. Econ. 16, 65–88 (1987). doi:10.1016/0304-4068(87)90022-X

            # D. Schmeidler, Subjective probability and expected utility without additivity.
            # Econometrica 57, 571 (1989). doi:10.2307/1911053

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
            if x[0] >= 0:
                coefficient = probability_alpha
            else:
                coefficient = probability_beta

            probability_a = x[1] ** coefficient / (
                x[1] ** coefficient + (1 - x[1]) ** coefficient
            ) ** (1 / coefficient)

            # compute probability of option B
            if x[2] >= 0:
                coefficient = probability_alpha
            else:
                coefficient = probability_beta

            probability_b = x[3] ** coefficient / (
                x[3] ** coefficient + (1 - x[3]) ** coefficient
            ) ** (1 / coefficient)

            expected_value_A = value_A * probability_a + rng.normal(0, added_noise)
            expected_value_B = value_B * probability_b + rng.normal(0, added_noise)

            # compute probability of choosing option A
            p_choose_A = np.exp(expected_value_A / choice_temperature) / (
                np.exp(expected_value_A / choice_temperature)
                + np.exp(expected_value_B / choice_temperature)
            )

            Y[idx] = p_choose_A

        experiment_data = pd.DataFrame(conditions)
        experiment_data.columns = [v.name for v in variables.independent_variables]
        experiment_data[variables.dependent_variables[0].name] = Y
        return experiment_data

    ground_truth = partial(run, added_noise=0.0)

    def domain():
        v_a = variables.independent_variables[0].allowed_values
        p_a = variables.independent_variables[1].allowed_values
        v_b = variables.independent_variables[2].allowed_values
        p_b = variables.independent_variables[3].allowed_values

        X = np.array(np.meshgrid(v_a, p_a, v_b, p_b)).T.reshape(-1, 4)
        return X

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

            y = ground_truth(X)[variables.dependent_variables[0].name]
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

        x_limit = [0, variables.independent_variables[1].value_range[1]]
        y_limit = [0, 1]
        x_label = "Probability of Choosing Option A"
        y_label = "Probability of Obtaining V(A)"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=2, fontsize="medium")
        plt.title(name, fontsize="x-large")

    collection = SyntheticExperimentCollection(
        name=name,
        description=prospect_theory.__doc__,
        params=params,
        variables=variables,
        domain=domain,
        run=run,
        ground_truth=ground_truth,
        plotter=plotter,
        factory_function=prospect_theory,
    )
    return collection
