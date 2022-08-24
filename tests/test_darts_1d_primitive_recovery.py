#!/usr/bin/env python
from typing import Callable, Sequence

import numpy as np
import pytest  # noqa:F401
import torch.nn
from matplotlib import pyplot as plt
from skl.darts import DARTSRegressor

non_interchangeable_primitives = [
    "none",
    "add",
    "subtract",
    "mult",
    "sigmoid",
    "exp",
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


def transform_through_primitive_cos(x: np.ndarray):
    y = np.cos(x)
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


def init_weights_uniform(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, -1, 1)


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
        max_epochs=300,
        arch_updates_per_epoch=1,
        param_weight_decay=3e-4,
        arch_weight_decay_df=0.05,
        arch_weight_decay=1e-4,
        arch_learning_rate_max=0.3,
        param_learning_rate_max=0.0025,
        param_learning_rate_min=0.01,
        param_momentum=0.90,
        train_classifier_coefficients=False,
        train_classifier_bias=False,
        primitives=primitives,
        # init_weights_function=init_weights_uniform,
    )

    regressor.fit(X, y)

    if verbose:
        y_predict = regressor.predict(X)
        print("\n", np.column_stack((X, y, regressor.predict(X))).round(2))
        print(get_primitive_from_single_node_model(regressor.model_))

        print(
            "Weight of winning primitive: {0}".format(
                regressor.model_[0]._arch_parameters[0][
                    0,
                    non_interchangeable_primitives.index(
                        get_primitive_from_single_node_model(regressor.model_)
                    ),
                ]
            )
        )
        print(
            "Weight of correct primitive: {0}".format(
                regressor.model_[0]._arch_parameters[0][
                    0, non_interchangeable_primitives.index(expected_primitive)
                ]
            )
        )
        print(regressor.model_[0]._arch_parameters[0].data)

        plt.plot(X, y, "o")
        plt.plot(X, y_predict, "-")
        plt.show()

    if expected_primitive == "add" or expected_primitive == "subtract":
        assert (
            get_primitive_from_single_node_model(regressor.model_) == expected_primitive
            or get_primitive_from_single_node_model(regressor.model_) == "mult"
        )
    else:
        assert (
            get_primitive_from_single_node_model(regressor.model_) == expected_primitive
        )


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


def test_primitive_fitting_mult():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_mult,
        "mult",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_relu():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_relu,
        "relu",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_exp():
    run_test_primitive_fitting(
        generate_x(start=-1, stop=1),
        transform_through_primitive_exp,
        "exp",
        primitives=non_interchangeable_primitives,
    )


def test_primitive_fitting_sigmoid():
    run_test_primitive_fitting(
        generate_x(start=-5, stop=5, num=500),
        transform_through_primitive_sigmoid,
        "sigmoid",
        primitives=non_interchangeable_primitives,
    )


# currently not working
def test_primitive_fitting_cos():
    run_test_primitive_fitting(
        generate_x(start=-4, stop=4),
        transform_through_primitive_cos,
        "cos",
        primitives=non_interchangeable_primitives,
    )


# currently not working
def test_primitive_fitting_softplus():
    run_test_primitive_fitting(
        generate_x(start=-4, stop=4, num=500),
        transform_through_primitive_softplus,
        "softplus",
        primitives=non_interchangeable_primitives,
    )


# currently not working
def test_primitive_fitting_softminus():
    run_test_primitive_fitting(
        generate_x(start=-4, stop=4, num=500),
        transform_through_primitive_softminus,
        "softminus",
        primitives=non_interchangeable_primitives,
    )


# currently not working
def test_primitive_fitting_inverse():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_inverse,
        "1/x",
        primitives=non_interchangeable_primitives,
    )


# currently not working
def test_primitive_fitting_ln():
    run_test_primitive_fitting(
        generate_x_log(-5, 5),
        transform_through_primitive_ln,
        "ln",
        primitives=non_interchangeable_primitives,
    )
