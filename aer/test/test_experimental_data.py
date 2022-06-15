import unittest

from experimental_data import ExperimentalData, load_experimental_data

from aer.variable import Variable, VariableCollection


class TestLoadExperimentalData(unittest.TestCase):
    def test_load_experimental_data(self):
        m = VariableCollection(
            independent_variables=[Variable("S1"), Variable("S2")],
            dependent_variables=[Variable("difference_detected")],
        )
        e = load_experimental_data("aer/test/data/weber_test_data.csv", m)
        self.assertIsInstance(e, ExperimentalData)
        print(e)


if __name__ == "__main__":
    unittest.main()
