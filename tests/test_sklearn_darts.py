import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from autora.skl.darts import DARTSRegressor, DARTSType, ValueType


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


class TestDarts(unittest.TestCase):
    def assertBetween(self, a, bmin, bmax):
        self.assertGreater(a, bmin)
        self.assertLess(a, bmax)

    def test_constant_model(self):

        X, y, const, epsilon = generate_noisy_constant_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        estimator = DARTSRegressor(num_graph_nodes=1)

        estimator.fit(X_train, y_train)

        self.assertIsNotNone(estimator)

        for y_pred_i in np.nditer(estimator.predict(X_test)):
            self.assertBetween(
                y_pred_i, const - (5.0 * epsilon), const + (5.0 * epsilon)
            )

        print(estimator.network_)

    def test_enum_string_inputs(self):

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
        DARTSRegressor(output_type=ValueType.PROBABILITY_DISTRIBUTION, **kwargs).fit(
            X, y
        )
        self.assertRaises(
            NotImplementedError, DARTSRegressor(output_type="class", **kwargs).fit, X, y
        )
        self.assertRaises(
            NotImplementedError,
            DARTSRegressor(output_type=ValueType.CLASS, **kwargs).fit,
            X,
            y,
        )

    def test_metaparam_optimization(self):

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
            self.assertBetween(y_pred_i, const - 0.01, const + 0.01)

        print(estimator.predict(X_test))


if __name__ == "__main__":
    unittest.main()
