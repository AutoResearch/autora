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

        X = data[["x1"]]
        y = data[["y"]]

        estimator = DARTS(
            VariableCollection(
                independent_variables=[Variable("x1")],
                dependent_variables=[Variable("y")],
            )
        )

        theory = estimator.fit(X, y)

        self.assertIsNotNone(theory)


if __name__ == "__main__":
    unittest.main()
