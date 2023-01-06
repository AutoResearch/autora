#!/usr/bin/env python
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401

from autora.skl.darts import DARTSRegressor

non_interchangeable_primitives = [
    "none",
    "add",
    "subtract",
    "logistic",
    "exp",
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


def transform_through_primitive_none(x: np.ndarray) -> np.ndarray:
    return x * 0.0


def transform_through_primitive_add(x: np.ndarray) -> np.ndarray:
    return x


def transform_through_primitive_subtract(x: np.ndarray) -> np.ndarray:
    return -x


def transform_through_primitive_relu(x: np.ndarray):
    y = x.copy()
    y[x < 0.0] = 0.0
    return y


def transform_through_primitive_sigmoid(x: np.ndarray):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def transform_through_primitive_exp(x: np.ndarray):
    y = np.exp(x)
    return y


def transform_through_primitive_cos(x: np.ndarray):
    y = np.cos(x)
    return y


def transform_through_primitive_cosplusx(x: np.ndarray):
    y = np.cos(x)
    return y + x


def transform_through_primitive_sin(x: np.ndarray):
    y = np.sin(x)
    return y


def transform_through_primitive_tanh(x: np.ndarray):
    y = np.tanh(x)
    return y


def transform_through_primitive_softplus(x: np.ndarray, beta=1.0):
    y = np.log(1 + np.exp(beta * x)) / beta
    return y


def transform_through_primitive_softminus(x: np.ndarray, beta=1.0):
    y = x - np.log(1 + np.exp(beta * x)) / beta
    return y


def transform_through_primitive_inverse(x: np.ndarray):
    y = 1.0 / x
    return y


def transform_through_primitive_ln(x: np.ndarray):
    y = np.log(x)
    return y


def transform_through_primitive_mult(x: np.ndarray, coefficient=5.0):
    y = coefficient * x
    return y


def get_primitive_from_single_node_model(model):
    primitive = model[0].primitives[np.argmax(model[0].max_alphas_normal()).numpy()]
    return primitive


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    expected_primitive: str,
    primitives: Sequence[str],
    verbose: bool = True,
):
    y = transformer(X)
    regressor = DARTSRegressor(
        num_graph_nodes=1,
        param_updates_per_epoch=100,
        max_epochs=400,
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
    )
    # print("XX")
    # print(X)
    # print("yy")
    # print(y)
    regressor.fit(X, y)

    if verbose:
        y_predict = regressor.predict(X)
        # print("y_pred")
        # print(y_predict)
        report_weights(X, expected_primitive, primitives, regressor, y)
        plot_results(X, y, y_predict, expected_primitive, primitives)

    assert get_primitive_from_single_node_model(regressor.model_) == expected_primitive


def plot_results(X, y, y_predict, expected_primitive, primitives):
    plt.plot(X, y, "o")
    plt.plot(X, y_predict, "-")
    # primitives_string = " ".join(primitives)
    # title = "expected " + expected_primitive + " from: " + primitives_string
    # plt.title(title)
    plt.savefig("tanh.png")
    plt.show()


def report_weights(X, expected_primitive, primitives, regressor, y):
    print("\n", np.column_stack((X, y, regressor.predict(X))).round(2))
    print(get_primitive_from_single_node_model(regressor.model_))
    print(
        "Weight of winning primitive: {0}".format(
            regressor.model_[0]._arch_parameters[0][
                0,
                primitives.index(
                    get_primitive_from_single_node_model(regressor.model_)
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


def test_primitive_fitting_restricted_none():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_none,
        "none",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_add():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_add,
        "add",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_subtract():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_subtract,
        "subtract",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_cos():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_cos,
        "cos",
        primitives=["cos", "sin", "subtract"],
    )


def test_primitive_fitting_none():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_none,
        "none",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_add():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_add,
        "add",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_subtract():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_subtract,
        "subtract",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_relu():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_relu,
        "relu",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_sigmoid():
    run_test_primitive_fitting(
        generate_x(-10, +10),
        transform_through_primitive_sigmoid,
        "logistic",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_exp():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_exp,
        "exp",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_cos():
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_cos,
        "cos",
        primitives=non_interchangeable_primitives,
    )


# def test_primitive_fitting_cosplusx():
#     run_test_primitive_fitting(
#         generate_x(start=0, stop=2 * np.pi),
#         transform_through_primitive_cosplusx,
#         "cos",
#         primitives=non_interchangeable_primitives,
#     )


def test_primitive_fitting_sin():
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_sin,
        "sin",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_tanh():
    run_test_primitive_fitting(
        generate_x(start=0, stop=2 * np.pi),
        transform_through_primitive_tanh,
        "tanh",
        primitives=non_interchangeable_primitives,
    )
