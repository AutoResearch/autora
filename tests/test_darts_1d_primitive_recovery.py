#!/usr/bin/env python
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401
from skl.darts import DARTSRegressor

non_interchangeable_primitives = [
    "none",
    "add",
    "subtract",
    "sigmoid",
    "exp",
    "1/x",
    "ln",
    "relu",
    "softplus",
    "softminus",
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
        max_epochs=100,
        arch_updates_per_epoch=100,
        primitives=primitives,
    )
    regressor.fit(X, y)

    if verbose:
        y_predict = regressor.predict(X)
        report_weights(X, expected_primitive, primitives, regressor, y)
        plot_results(X, y, y_predict)

    assert get_primitive_from_single_node_model(regressor.model_) == expected_primitive


def plot_results(X, y, y_predict):
    plt.plot(X, y, "o")
    plt.plot(X, y_predict, "-")
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
        generate_x(),
        transform_through_primitive_sigmoid,
        "sigmoid",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_exp():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_exp,
        "exp",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_inverse():
    run_test_primitive_fitting(
        generate_x_log(0, +1.1),
        transform_through_primitive_inverse,
        "1/x",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_ln():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_ln,
        "ln",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_softplus():
    run_test_primitive_fitting(
        generate_x(start=-4, stop=4, num=500),
        transform_through_primitive_softplus,
        "softplus",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_softminus():
    run_test_primitive_fitting(
        generate_x(start=-4, stop=4, num=500),
        transform_through_primitive_softminus,
        "softminus",
        primitives=non_interchangeable_primitives,
    )
