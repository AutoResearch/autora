import numpy as np

from autora.model.inventory import register
from .expected_value import expected_value_theory_metadata

# general meta parameters
added_noise = 0.00

# prospect theory parameters
# parameters taken from:
# D. Kahneman, A. Tversky, Prospect theory: An analysis of decision under risk. Econometrica
# 47, 263–292 (1979). doi:10.2307/1914185
prospect_theory_choice_temperature = 0.1
prospect_theory_value_alpha = 0.88
prospect_theory_value_beta = 0.88
prospect_theory_value_lambda = 2.25
prospect_theory_probability_alpha = 1.0  # 0.61
prospect_theory_probability_beta = 0.69

# prospect theory

prospect_theory_metadata = expected_value_theory_metadata


def prospect_theory_experiment(
    X: np.ndarray,
    choice_temperature: float = prospect_theory_choice_temperature,
    value_alpha=prospect_theory_value_alpha,
    value_beta=prospect_theory_value_beta,
    value_lambda=prospect_theory_value_lambda,
    probability_alpha=prospect_theory_probability_alpha,
    probability_beta=prospect_theory_probability_beta,
    std=added_noise,
):

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

        expected_value_A = value_A * probability_a + np.random.normal(0, std)
        expected_value_B = value_B * probability_b + np.random.normal(0, std)

        # compute probability of choosing option A
        p_choose_A = np.exp(expected_value_A / choice_temperature) / (
            np.exp(expected_value_A / choice_temperature)
            + np.exp(expected_value_B / choice_temperature)
        )

        Y[idx] = p_choose_A

    return Y


def prospect_theory_data(metadata):

    v_a = metadata.independent_variables[0].allowed_values
    p_a = metadata.independent_variables[1].allowed_values
    v_b = metadata.independent_variables[2].allowed_values
    p_b = metadata.independent_variables[3].allowed_values

    X = np.array(np.meshgrid(v_a, p_a, v_b, p_b)).T.reshape(-1, 4)

    y = prospect_theory_experiment(X, std=0)

    return X, y


def plot_prospect_theory(model=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    metadata = prospect_theory_metadata()

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

        y = prospect_theory_experiment(X, std=0)
        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())
        plt.plot(p_a, y, label=f"$V(A) = {v_a}$ (Original)", c=colors[col_keys[idx]])
        if model is not None:
            y = model.predict(X)
            plt.plot(p_a, y, label=f"$V(A) = {v_a}$ (Recovered)", c=colors[col_keys[idx]], linestyle="--")

    x_limit = [0, metadata.independent_variables[1].value_range[1]]
    y_limit = [0, 1]
    x_label = "Probability of Choosing Option A"
    y_label = "Probability of Obtaining V(A)"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="medium")
    plt.title("Prospect Theory", fontsize="x-large")
    plt.show()


# plot_prospect_theory()

# value_alpha = 0.88
# value_beta = 0.88
# value_lambda = 2.25
# probability_alpha = 1.0 #0.61
# probability_beta = 0.69
#
# x_range = np.linspace(0, 1, 1000)
# y = np.zeros((x_range.shape[0],1))
#
# for idx, x in enumerate(x_range):
#
#     # if x >= 0:
#     #     value_A = x ** value_alpha
#     # else:
#     #     value_A = - value_lambda * (-x) ** (value_beta)
#
#     if x >= 0:
#         coefficient = probability_alpha
#     else:
#         coefficient = probability_beta
#
#     probability_a = x ** coefficient / \
#                     (x ** coefficient + (1 - x) ** coefficient) ** (1 / coefficient)
#
#     y[idx] = probability_a
#
# import matplotlib.pyplot as plt
# plt.plot(x_range, y)
# plt.show()

register(
    "prospect_theory",
    metadata=prospect_theory_metadata,
    data=prospect_theory_data,
    synthetic_experiment_runner=prospect_theory_experiment,
    name="Prospect Theory",
)
