from typing import Callable, Optional

import numpy as np
import pytest  # noqa: 401

from autora.skl.bms import BMSRegressor


def generate_x_2d(start=-1, stop=1, num=40):
    step = abs(stop - start) / num
    x2 = np.mgrid[start:stop:step, start:stop:step].reshape(2, -1).T
    return x2


def transform_through_primitive_softmax_2d(x: np.ndarray) -> np.ndarray:
    return np.array([y for y in [softmax_2d(x, y) for x, y in x]])


def run_test_primitive_fitting_2d(
    X: np.ndarray,
    transformer: Callable,
    verbose: bool = False,
    custom_primitive: Optional[Callable] = None,
    root=None,
):
    y = transformer(X)
    regressor = BMSRegressor(epochs=30)
    regressor.fit(X, y.ravel(), custom_ops=[custom_primitive], root=root)
    if verbose:
        print(regressor.model_)
        print(regressor.pms.trees)


def softmax_2d(x, y):
    return np.exp(x) / (np.exp(x) + np.exp(y))


def test_primitive_fitting_softmax_2d_fixed_root():
    run_test_primitive_fitting_2d(
        generate_x_2d(),
        transformer=transform_through_primitive_softmax_2d,
        custom_primitive=softmax_2d,
        root=softmax_2d,
        verbose=True,
    )
