import unittest

from aer_bms.mcmc import Tree


class TestMCMC(unittest.TestCase):
    def test_trees_as_strings(
        self, string="(P120 + (((ALPHACAT / _a2) + (_a2 * CDH3)) + _a0))"
    ):
        # Create the formula
        prior_par = {
            "Nopi_/": 0,
            "Nopi_cosh": 0,
            "Nopi_-": 0,
            "Nopi_sin": 0,
            "Nopi_tan": 0,
            "Nopi_tanh": 0,
            "Nopi_**": 0,
            "Nopi_pow2": 0,
            "Nopi_pow3": 0,
            "Nopi_exp": 0,
            "Nopi_log": 0,
            "Nopi_sqrt": 0,
            "Nopi_cos": 0,
            "Nopi_sinh": 0,
            "Nopi_abs": 0,
            "Nopi_+": 0,
            "Nopi_*": 0,
            "Nopi_fac": 0,
        }

        t = Tree(prior_par=prior_par, from_string=string)
        for i in range(10000):
            t.mcmc_step(verbose=True)
            print("-" * 150)
            t2 = Tree(from_string=str(t))
            self.assertEqual(str(t2), str(t))
            print(t)
            print(t2)
        return t


if __name__ == "__main__":
    unittest.main()
