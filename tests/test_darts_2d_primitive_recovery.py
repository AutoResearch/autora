#!/usr/bin/env python
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401
from more_itertools import powerset

from autora.skl.darts import DARTSRegressor

non_interchangeable_primitives = [
    "none",
    "add",
    "subtract",
    "logistic",
    # "exp",
    "relu",
    "cos",
    "sin",
    "tanh",
]


def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def generate_x_log(start=-1, stop=1, num=500, base=10):
    x = np.expand_dims(np.logspace(start=start, stop=stop, num=num, base=base), 1)
    return x


def get_2d_primitive_from_two_node_model(model):
    # print(model[0].max_alphas_normal())
    # print(model[0].max_betas_normal())
    # print(model)
    print("LEN", len(model[0].max_betas_normal()))
    for i in range(len(model[0].max_betas_normal())):
        num_prev_nodes = 1 + i
        set = np.arange(num_prev_nodes)
        subsets = powerset(set)
        subset_idx = np.argmax(model[0].max_betas_normal()[i]).numpy()
        print("subset index", subset_idx)
        # subset = subsets[subset_idx + 1]
        # print("THE SUBSET", subset)
        for (index, subset) in enumerate(subsets):
            print(index, subset)
            if index == subset_idx + 1:
                ans_subset = subset
                # print("ANS SUBSET", ans_subset)
        found_primitives = []
        for j in range(len(ans_subset)):
            edge_idx = ans_subset[j]
            # print(edge_idx)
            primitive_idx = np.argmax(model[0].max_alphas_normal()[edge_idx]).numpy()
            print("CURR PRIMITIVE", model[0].primitives[primitive_idx])
            found_primitives.append(model[0].primitives[primitive_idx])

    print("FOUND PRIMITIVES", found_primitives)
    return found_primitives

    # primitive1 = model[0].primitives[np.argmax(model[0].max_alphas_normal()).numpy()]
    # primitive2 = model[1].primitives[np.argmax(model[0].max_alphas_normal()).numpy()]
    # return primitive1, primitive2


def transform_through_primitive_xcos(x: np.ndarray):
    y = np.cos(x)
    return x * y


def transform_through_primitive_cosxsquared(x: np.ndarray):
    y = np.cos(x * x)
    return y


def transform_through_primitive_xcospluscos(x: np.ndarray):
    y = np.cos(x)
    return x * y + y


def transform_through_primitive_xtanh(x: np.ndarray):
    y = np.tanh(x)
    return x * y


def transform_through_primitive_xrelu(x: np.ndarray):
    y = x.copy()
    y[x < 0.0] = 0.0
    return x * y


def transform_through_primitive_xcosplusx(x: np.ndarray):
    y = np.cos(x)
    return x * y + x


def transform_through_primitive_cosplussin(x: np.ndarray):
    y = np.cos(x)
    z = np.sin(x)
    return y + z


def transform_through_primitive_xsin(x: np.ndarray):
    y = np.sin(x)
    return x * y


def transform_through_primitive_xsquared(x: np.ndarray):
    return x * x


def transform_through_primitive_sincos(x: np.ndarray):
    y = np.sin(x)
    z = np.cos(x)
    return y * z


def transform_through_primitive_xsigmoid(x: np.ndarray):
    y = 1.0 / (1.0 + np.exp(-x)) * x
    return y


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    expected_primitive: Sequence[str],
    primitives: Sequence[str],
    verbose: bool = True,
):
    y = transformer(X)
    regressor = DARTSRegressor(
        num_graph_nodes=2,
        param_updates_per_epoch=1000,
        max_epochs=500,
        arch_updates_per_epoch=1,
        param_weight_decay=3e-4,
        arch_weight_decay_df=0.001,
        arch_weight_decay=1e-4,
        arch_learning_rate_max=0.3,
        param_learning_rate_max=0.0025,
        param_learning_rate_min=0.01,
        param_momentum=0.90,
        primitives=primitives,
        train_classifier_bias=False,
        train_classifier_coefficients=False,
        param_updates_for_sampled_model=1000,
    )
    regressor.fit(X, y)

    if verbose:
        y_predict = regressor.predict(X)
        # report_weights(X, expected_primitive, primitives, regressor, y)
        plot_results(X, y, y_predict, expected_primitive)

    found_primitives = get_2d_primitive_from_two_node_model(regressor.model_)
    print(found_primitives)

    assert (
        found_primitives == expected_primitive
        or list(reversed(found_primitives)) == expected_primitive
    )


def plot_results(X, y, y_predict, expected_primitive):
    plt.plot(X, y, "o")
    plt.plot(X, y_predict, "-")
    # plt.title(str(expected_primitive))
    plt.savefig("xtanh.png")
    plt.show()


def report_weights(X, expected_primitive, primitives, regressor, y):
    print("\n", np.column_stack((X, y, regressor.predict(X))).round(2))
    print(get_2d_primitive_from_two_node_model(regressor.model_))
    print(
        "Weight of winning primitive: {0}".format(
            regressor.model_[0]._arch_parameters[0][
                0,
                primitives.index(
                    get_2d_primitive_from_two_node_model(regressor.model_)
                ),
            ]
        )
    )
    print(
        "Weight of correct primitive: {0}".format(
            regressor.model_[0]._arch_parameters[0][
                0, primitives.index(expected_primitive)
            ]
        )
    )
    print(regressor.model_[0]._arch_parameters[0].data)


def test_primitive_fitting_xsquared():  # x * x
    run_test_primitive_fitting(
        generate_x(start=-10, stop=10),
        transform_through_primitive_xsquared,
        expected_primitive=["add", "add"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_cosxsquared():  # cos(x * x)
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_cosxsquared,
        expected_primitive=["add", "cos"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_xcos():  # x * cos(x)
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_xcos,
        expected_primitive=["add", "cos"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_xcospluscos():  # x * cos(x)
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_xcospluscos,
        expected_primitive=["add", "cos"],
        primitives=non_interchangeable_primitives,
    )


# def test_primitive_fitting_xcosplusx():  # x * cos(x) + x
#     run_test_primitive_fitting(
#         generate_x(start=0, stop=5 * np.pi),
#         transform_through_primitive_xcosplusx,
#         "xcosplusx",
#         primitives=non_interchangeable_primitives,
#     )


def test_primitive_fitting_xsin():  # x * sin(x)
    run_test_primitive_fitting(
        generate_x(start=0, stop=3 * np.pi),
        transform_through_primitive_xsin,
        expected_primitive=["add", "sin"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_xtanh():  # x * tanh
    run_test_primitive_fitting(
        generate_x(start=-3, stop=3),
        transform_through_primitive_xtanh,
        expected_primitive=["add", "tanh"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_xrelu():  # x * relu
    run_test_primitive_fitting(
        generate_x(start=-20, stop=50),
        transform_through_primitive_xrelu,
        expected_primitive=["add", "relu"],
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_xsigmoid():  # x * sigmoid
    run_test_primitive_fitting(
        generate_x(start=-5, stop=2),
        transform_through_primitive_xsigmoid,
        expected_primitive=["add", "logistic"],
        primitives=non_interchangeable_primitives,
    )


# def test_primitive_fitting_sincos():  # sin(x) * cos(x)
#     run_test_primitive_fitting(
#         generate_x(start=0, stop=2 * np.pi),
#         transform_through_primitive_sincos,
#         expected_primitive=['sin', 'cos'],
#         primitives=non_interchangeable_primitives,
#     )
