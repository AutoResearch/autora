import numpy as np
import pytest
from skl.darts_execution_monitor import create_basic_execution_monitor
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


def test_execution_monitor():
    import matplotlib.pyplot as plt

    X, y, const, epsilon = generate_noisy_constant_data()

    kwargs = dict()

    execution_monitor_0, display_0 = create_basic_execution_monitor()

    DARTSRegressor(
        primitives=["add", "subtract", "none", "mult", "sigmoid"],
        execution_monitor=execution_monitor_0,
        num_graph_nodes=3,
        max_epochs=100,
        param_updates_per_epoch=100,
        **kwargs
    ).fit(X, y)
    display_0()

    execution_monitor_1, display_1 = create_basic_execution_monitor()
    DARTSRegressor(
        primitives=["add", "ln"],
        num_graph_nodes=5,
        max_epochs=100,
        param_updates_per_epoch=100,
        execution_monitor=execution_monitor_1,
        **kwargs
    ).fit(X, y)
    display_1()

    plt.show()
