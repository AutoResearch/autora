#!/usr/bin/env python
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401

from autora.skl.bms import BMSRegressor

warnings.filterwarnings("ignore")


def generate_x_2d(start=-1, stop=1, num=40):
    step = abs(stop - start) / num
    x2 = np.mgrid[start:stop:step, start:stop:step].reshape(2, -1).T
    return x2


def generate_pos_x_2d(start=0.5, stop=1, num=40):
    step = abs(stop - start) / num
    x2 = np.mgrid[start:stop:step, start:stop:step].reshape(2, -1).T
    return x2


def transform_through_primitive_add_2d(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1]


def transform_through_primitive_subtract_2d(x: np.ndarray) -> np.ndarray:
    return x[:, 0] - x[:, 1]


def transform_through_primitive_mult_2d(x: np.ndarray) -> np.ndarray:
    return np.multiply(x[:, 0], x[:, 1])


def transform_through_primitive_div_2d(x: np.ndarray) -> np.ndarray:
    return np.multiply(x[:, 0], np.reciprocal(x[:, 1]))


def transform_through_primitive_pow_2d(x: np.ndarray) -> np.ndarray:
    return np.power(x[:, 0], x[:, 1])


def run_test_primitive_fitting_2d(
    X: np.ndarray,
    transformer: Callable,
    verbose: bool = True,
):
    y = transformer(X)
    regressor = BMSRegressor(epochs=30)
    regressor.fit(X, y.ravel())
    if verbose:
        y_predict = regressor.predict(X)
        plot_results_2d(X, y, y_predict)
        print(regressor.model_)
        print(regressor.pms.trees)


def plot_results_2d(X, y, y_predict):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, s=0.5)
    n = np.sqrt(X.shape[0]).astype(int)
    ax.plot_surface(
        X[:, 0].reshape(n, n),
        X[:, 1].reshape(n, n),
        y_predict.reshape(n, n),
        color="orange",
    )
    plt.show()


def test_primitive_fitting_add_2d():
    run_test_primitive_fitting_2d(
        generate_x_2d(),
        transform_through_primitive_add_2d,
    )


def test_primitive_fitting_subtract_2d():
    run_test_primitive_fitting_2d(
        generate_x_2d(),
        transform_through_primitive_subtract_2d,
    )


def test_primitive_fitting_mult_2d():
    run_test_primitive_fitting_2d(
        generate_x_2d(),
        transform_through_primitive_mult_2d,
    )


def test_primitive_fitting_div_2d():
    run_test_primitive_fitting_2d(
        generate_pos_x_2d(),
        transform_through_primitive_div_2d,
    )


def test_primitive_fitting_pow_2d():
    run_test_primitive_fitting_2d(
        generate_pos_x_2d(),
        transform_through_primitive_pow_2d,
    )
