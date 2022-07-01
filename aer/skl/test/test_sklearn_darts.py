import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from aer.object_of_study import Variable, VariableCollection
from aer.skl.darts import DARTS


def generate_constant_data(
    const: float = 0.5, epsilon: float = 0.01, num: int = 1000, seed: int = 42
):
    X = np.expand_dims(np.linspace(start=0, stop=1, num=num), 1)
    y = np.random.default_rng(seed).normal(loc=const, scale=epsilon, size=num)
    variable_collection = VariableCollection(
        independent_variables=[Variable("x")],
        dependent_variables=[Variable("y")],
    )
    return X, y, variable_collection


class TestDarts(unittest.TestCase):
    def assertBetween(self, a, bmin, bmax):
        self.assertGreater(a, bmin)
        self.assertLess(a, bmax)

    def test_constant_model(self):

        const = 0.5
        epsilon = 0.01

        X, y, variable_collection = generate_constant_data(const, epsilon)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        estimator = DARTS(variable_collection, num_graph_nodes=1)

        estimator.fit(X_train, y_train)

        self.assertIsNotNone(estimator)

        for y_pred_i in np.nditer(estimator.predict(X_test)):
            self.assertBetween(
                y_pred_i, const - (5.0 * epsilon), const + (5.0 * epsilon)
            )

        print(estimator.network_)

    def test_metaparam_optimization(self):

        X, y, variable_collection = generate_constant_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        estimator = GridSearchCV(
            estimator=DARTS(variable_collection),
            param_grid=[{"num_graph_nodes": [1, 2]}],
        )

        estimator.fit(X_train, y_train)

        self.assertIsNotNone(estimator)

        print(estimator.predict(X_test))


if __name__ == "__main__":
    unittest.main()
