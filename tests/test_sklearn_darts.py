from typing import Callable, Sequence

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, train_test_split

from autora.skl.darts import PRIMITIVES, DARTSRegressor, DARTSType, ValueType


def generate_noisy_constant_data(
    const: float = 0.5, epsilon: float = 0.01, num: int = 1000, seed: int = 42
):
    X = np.expand_dims(np.linspace(start=0, stop=1, num=num), 1)
    y = np.random.default_rng(seed).normal(loc=const, scale=epsilon, size=num)
    return X, y, const, epsilon


def generate_constant_data(const: float = 0.5, num: int = 1000):
    X = np.expand_dims(np.linspace(start=0, stop=1, num=num), 1)
    y = const * np.ones(num)
    return X, y, const


def transform_through_primitive_none(x: np.ndarray) -> np.ndarray:
    return x * 0.0


def transform_through_primitive_add(x: np.ndarray) -> np.ndarray:
    return x


def transform_through_primitive_subtract(x: np.ndarray) -> np.ndarray:
    return -x


def transform_through_primitive_relu(x: np.ndarray):
    y = x.copy()
    y[x < 0] = 0.0
    return y


def get_primitive_from_single_node_model(model):
    primitive = model[0].primitives[np.argmax(model[0].max_alphas_normal()).numpy()]
    return primitive


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    expected_primitive: str,
    primitives: Sequence[str] = PRIMITIVES,
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
    assert get_primitive_from_single_node_model(regressor.model_) == expected_primitive


def test_primitive_fitting_restricted_none():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(
        X,
        transform_through_primitive_none,
        "none",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_add():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(
        X,
        transform_through_primitive_add,
        "add",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_subtract():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(
        X,
        transform_through_primitive_subtract,
        "subtract",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_none():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(X, transform_through_primitive_none, "none")


def test_primitive_fitting_lin_relu():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(X, transform_through_primitive_relu, "lin_relu")


def test_primitive_fitting_add():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(X, transform_through_primitive_add, "add")


def test_primitive_fitting_subtract():
    X = np.expand_dims(np.linspace(start=-1, stop=1, num=100), 1)
    run_test_primitive_fitting(X, transform_through_primitive_subtract, "subtract")


def test_constant_model():

    X, y, const, epsilon = generate_noisy_constant_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimator = DARTSRegressor(num_graph_nodes=1)

    estimator.fit(X_train, y_train)

    assert estimator is not None

    for y_pred_i in np.nditer(estimator.predict(X_test)):
        (const - (5.0 * epsilon)) <= y_pred_i <= (const + (5.0 * epsilon))

    print(estimator.network_)


def test_enum_string_inputs():

    X, y, const, epsilon = generate_noisy_constant_data()

    kwargs = dict(
        num_graph_nodes=1,
        max_epochs=1,
        arch_updates_per_epoch=1,
        param_updates_per_epoch=1,
    )

    DARTSRegressor(darts_type="fair", **kwargs).fit(X, y)
    DARTSRegressor(darts_type=DARTSType.FAIR, **kwargs).fit(X, y)
    DARTSRegressor(darts_type="original", **kwargs).fit(X, y)
    DARTSRegressor(darts_type=DARTSType.ORIGINAL, **kwargs).fit(X, y)

    DARTSRegressor(output_type="probability", **kwargs).fit(X, y)
    DARTSRegressor(output_type=ValueType.PROBABILITY, **kwargs).fit(X, y)
    DARTSRegressor(output_type=ValueType.PROBABILITY_SAMPLE, **kwargs).fit(X, y)
    DARTSRegressor(output_type="probability_distribution", **kwargs).fit(X, y)
    DARTSRegressor(output_type=ValueType.PROBABILITY_DISTRIBUTION, **kwargs).fit(X, y)
    with pytest.raises(NotImplementedError):
        DARTSRegressor(output_type="class", **kwargs).fit(X, y)
    with pytest.raises(NotImplementedError):
        DARTSRegressor(output_type=ValueType.CLASS, **kwargs).fit(X, y)


def test_primitive_selection():
    X, y, const, epsilon = generate_noisy_constant_data()

    kwargs = dict(
        num_graph_nodes=1,
        max_epochs=1,
        arch_updates_per_epoch=1,
        param_updates_per_epoch=1,
    )

    DARTSRegressor(primitives=["add", "subtract", "none"], **kwargs).fit(X, y)
    DARTSRegressor(primitives=PRIMITIVES, **kwargs).fit(X, y)
    with pytest.raises(KeyError):
        KeyError, DARTSRegressor(primitives=["doesnt_exist"], **kwargs).fit(X, y)


def test_metaparam_optimization():

    X, y, const = generate_constant_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimator = GridSearchCV(
        estimator=DARTSRegressor(),
        cv=2,
        param_grid=[
            {
                "max_epochs": [10, 50],
                "arch_updates_per_epoch": [5, 10, 15],
                "param_updates_per_epoch": [5, 10, 15],
                "num_graph_nodes": [1, 2, 3],
            }
        ],
    )

    estimator.fit(X_train, y_train)

    print(estimator.best_params_)
    print(X_test)
    print(estimator.predict(X_test))

    for y_pred_i in np.nditer(estimator.predict(X_test)):
        assert (const - 0.01) < y_pred_i < (const + 0.01)

    print(estimator.predict(X_test))
