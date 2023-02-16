import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: 401

from autora.skl.bsr import BSRRegressor

warnings.filterwarnings("ignore")


def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def generate_pos_x(start=0.5, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def generate_x_log(start=-1, stop=1, num=500, base=10):
    x = np.expand_dims(np.logspace(start=start, stop=stop, num=num, base=base), 1)
    return x


def transform_through_primitive_pow2(x: np.ndarray) -> np.ndarray:
    return x**2


def transform_through_primitive_pow3(x: np.ndarray) -> np.ndarray:
    return x**3


def transform_through_primitive_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(x)


def transform_through_primitive_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def transform_through_primitive_fac(x: np.ndarray) -> np.ndarray:
    y = []
    for x_i in x:
        y.append(np.math.gamma(x_i[0] + 1.0))
    y_hat = np.array(y)
    return np.expand_dims(y_hat, 1)


def transform_through_primitive_none(x: np.ndarray) -> np.ndarray:
    return x * 0


def transform_through_primitive_add(x: np.ndarray) -> np.ndarray:
    return x


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


def transform_through_primitive_cosh(x: np.ndarray):
    y = np.cosh(x)
    return y


def transform_through_primitive_sin(x: np.ndarray):
    y = np.sin(x)
    return y


def transform_through_primitive_sinh(x: np.ndarray):
    y = np.sinh(x)
    return y


def transform_through_primitive_tan(x: np.ndarray):
    y = np.tan(x)
    return y


def transform_through_primitive_tanh(x: np.ndarray):
    y = np.tanh(x)
    return y


def transform_through_primitive_inverse(x: np.ndarray):
    y = 1.0 / x
    return y


def transform_through_primitive_ln(x: np.ndarray):
    y = np.log(x)
    return y


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    verbose: bool = False,
):
    y = transformer(X)
    regressor = BSRRegressor(tree_num=3, itr_num=10, val=10)
    regressor.fit(X, y.ravel())
    if verbose:
        y_predict = regressor.predict(X)
        for x_i in X.T[
            :,
        ]:
            plot_results(x_i, y, y_predict)
        print(regressor._model())


def plot_results(X, y, y_predict):
    plt.figure()
    plt.plot(X, y, "o")
    plt.plot(X, y_predict, "-")
    plt.show()


def test_primitive_fitting_restricted_add():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_add,
    )


def test_primitive_fitting_pow2():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_pow2,
    )


def test_primitive_fitting_pow3():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_pow3,
    )


def test_primitive_fitting_sqrt():
    run_test_primitive_fitting(
        generate_pos_x(),
        transform_through_primitive_sqrt,
    )


def test_primitive_fitting_abs():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_abs,
    )


def test_primitive_fitting_fac():
    run_test_primitive_fitting(
        generate_pos_x(),
        transform_through_primitive_fac,
    )
