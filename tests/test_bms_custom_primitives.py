import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401

from autora.skl.bms import BMSRegressor
from autora.theorist.bms.prior import relu


def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def generate_x_2d(start=-1, stop=1, num=40):
    step = abs(stop - start) / num
    x2 = np.mgrid[start:stop:step, start:stop:step].reshape(2, -1).T
    return x2


def transform_through_primitive_relu(x: np.ndarray) -> np.ndarray:
    return np.array([y for y in [relu(x) for x in x]])


def transform_through_primitive_softmax_2d(x: np.ndarray) -> np.ndarray:
    return np.array([y for y in [softmax_2d(x, y) for x, y in x]])


def transform_through_primitive_softmax_1d(x: np.ndarray) -> np.ndarray:
    return np.array([y for y in [softmax_1d(x) for x in x]])


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    verbose: bool = False,
    custom_primitive: Optional[Callable] = None,
):
    y = transformer(X)
    regressor = BMSRegressor(epochs=30)
    regressor.fit(X, y.ravel(), custom_ops=[custom_primitive])
    # custom primitive is none or it is in bms' custom ops
    assert custom_primitive is None or custom_primitive in regressor.custom_ops.values()
    # custom primitive is none or its name is in bms' ops
    assert custom_primitive is None or custom_primitive.__name__ in regressor.ops.keys()
    # custom primitive is none or it is in the fitted models' custom ops
    assert (
        custom_primitive is None
        or custom_primitive in regressor.model_.custom_ops.values()
    )
    # custom primitive is none or its name is in the fitted models' ops
    assert (
        custom_primitive is None
        or custom_primitive.__name__ in regressor.model_.ops.keys()
    )
    if verbose:
        y_predict = regressor.predict(X)
        for x_i in X.T[
            :,
        ]:
            plot_results(x_i, y, y_predict)
        print(regressor.model_)
        print(regressor.pms.trees)


def plot_results(X, y, y_predict):
    plt.figure()
    plt.plot(X, y, "o")
    plt.plot(X, y_predict, "-")
    plt.show()


def run_test_primitive_fitting_2d(
    X: np.ndarray,
    transformer: Callable,
    verbose: bool = False,
    custom_primitive: Optional[Callable] = None,
):
    y = transformer(X)
    random.seed(180)
    regressor = BMSRegressor(epochs=30)
    regressor.fit(X, y.ravel(), custom_ops=[custom_primitive])
    # custom primitive is none or it is in bms' custom ops
    assert custom_primitive is None or custom_primitive in regressor.custom_ops.values()
    # custom primitive is none or its name is in bms' ops
    assert custom_primitive is None or custom_primitive.__name__ in regressor.ops.keys()
    # custom primitive is none or it is in the fitted models' custom ops
    assert (
        custom_primitive is None
        or custom_primitive in regressor.model_.custom_ops.values()
    )
    # custom primitive is none or its name is in the fitted models' ops
    assert (
        custom_primitive is None
        or custom_primitive.__name__ in regressor.model_.ops.keys()
    )
    if verbose:
        print(regressor.model_)
        print(regressor.pms.trees)


def test_primitive_fitting_relu():
    run_test_primitive_fitting(
        generate_x(), transform_through_primitive_relu, custom_primitive=relu
    )


def softmax_2d(x, y):
    return np.exp(x) / (np.exp(x) + np.exp(y))


def test_primitive_fitting_softmax_2d():
    run_test_primitive_fitting_2d(
        generate_x_2d(),
        transformer=transform_through_primitive_softmax_2d,
        custom_primitive=softmax_2d,
    )


def softmax_1d(x):
    return np.exp(x) / (np.exp(x))


def test_primitive_fitting_softmax_1d():
    run_test_primitive_fitting(
        generate_x(),
        transformer=transform_through_primitive_softmax_1d,
        custom_primitive=softmax_1d,
    )
