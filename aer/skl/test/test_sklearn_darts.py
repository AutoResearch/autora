import unittest

import numpy as np
import pandas as pd

from aer.object_of_study import Variable, VariableCollection
from aer.skl.darts import DARTS


class TestDarts(unittest.TestCase):
    def test_constant_model(self):

        const = 0.5
        epsilon = 0.01

        seed = 42
        num = 1000

        data = pd.DataFrame(
            {
                "x1": np.linspace(start=0, stop=1, num=num),
                "y": np.random.default_rng(seed).normal(
                    loc=const, scale=epsilon, size=num
                ),
            }
        )

        X_train = data[["x1"]]
        y_train = data[["y"]]

        estimator = DARTS(
            VariableCollection(
                independent_variables=[Variable("x1")],
                dependent_variables=[Variable("y")],
            )
        )

        estimator.fit(X_train, y_train)

        self.assertIsNotNone(estimator)

        X_test = np.expand_dims(np.linspace(start=0, stop=1, num=5), axis=1)
        y_pred = estimator.predict(X_test)

        places = np.ceil(np.log10(1.0 / epsilon)).astype(int)

        for y_pred_i in np.nditer(y_pred):
            self.assertAlmostEqual(y_pred_i, const, places)


if __name__ == "__main__":
    unittest.main()
