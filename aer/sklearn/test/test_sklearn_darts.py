import unittest

import numpy as np

from aer.sklearn.darts import DARTS


class TestDarts(unittest.TestCase):
    def test_constant_model(self):

        const = 0.5
        epsilon = 0.01

        seed = 42
        num = 1000

        X = np.linspace(start=0, stop=1, num=num)
        y = np.random.default_rng(seed).normal(loc=const, scale=epsilon, size=num)

        estimator = DARTS()

        theory = estimator.fit(X, y)

        self.assertIsNotNone(theory)


if __name__ == "__main__":
    unittest.main()
