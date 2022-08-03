import unittest

import numpy as np
import pandas as pd

from aer_bms.mcmc import Tree
from aer_bms.utils import plt


def test3(num_points=10, samples=100000):
    # Create the data
    x = pd.DataFrame(
        dict([("x%d" % i, np.random.uniform(0, 10, num_points)) for i in range(5)])
    )
    eps = np.random.normal(0.0, 5, num_points)
    y = 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3 + eps
    x.to_csv("data_x.csv", index=False)
    y.to_csv("data_y.csv", index=False, header=["y"])

    # Create the formula
    prior_par = {
        "Nopi_/": 5.912205942815285,
        "Nopi_cosh": 8.12720511103694,
        "Nopi_-": 3.350846072163632,
        "Nopi_sin": 5.965917796154835,
        "Nopi_tan": 8.127427922862411,
        "Nopi_tanh": 7.799259068142255,
        "Nopi_**": 6.4734429542245495,
        "Nopi_pow2": 3.3017352779079734,
        "Nopi_pow3": 5.9907496760026175,
        "Nopi_exp": 4.768665265735502,
        "Nopi_log": 4.745957377206544,
        "Nopi_sqrt": 4.760686909134266,
        "Nopi_cos": 5.452564657261127,
        "Nopi_sinh": 7.955723540761046,
        "Nopi_abs": 6.333544134938385,
        "Nopi_+": 5.808163661224514,
        "Nopi_*": 5.002213595420244,
        "Nopi_fac": 10.0,
        "Nopi2_*": 1.0,
    }
    t = Tree(
        variables=["x%d" % i for i in range(5)],
        parameters=["a%d" % i for i in range(10)],
        x=x,
        y=y,
        prior_par=prior_par,
        BT=1.0,
    )
    # MCMC
    t.mcmc(burnin=2000, thin=10, samples=samples, verbose=True)

    # Predict
    print(t.predict(x))
    print(y)
    print(50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)

    plt.plot(t.predict(x), 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)
    plt.show()

    return t


def test4(num_points=10, samples=1000):
    # Create the data
    x = pd.DataFrame(
        dict([("x%d" % i, np.random.uniform(0, 10, num_points)) for i in range(5)])
    )
    eps = np.random.normal(0.0, 5, num_points)
    y = 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3 + eps
    x.to_csv("data_x.csv", index=False)
    y.to_csv("data_y.csv", index=False, header=["y"])

    xtrain, ytrain = x.iloc[5:], y.iloc[5:]
    xtest, ytest = x.iloc[:5], y.iloc[:5]

    # Create the formula
    prior_par = {
        "Nopi_/": 5.912205942815285,
        "Nopi_cosh": 8.12720511103694,
        "Nopi_-": 3.350846072163632,
        "Nopi_sin": 5.965917796154835,
        "Nopi_tan": 8.127427922862411,
        "Nopi_tanh": 7.799259068142255,
        "Nopi_**": 6.4734429542245495,
        "Nopi_pow2": 3.3017352779079734,
        "Nopi_pow3": 5.9907496760026175,
        "Nopi_exp": 4.768665265735502,
        "Nopi_log": 4.745957377206544,
        "Nopi_sqrt": 4.760686909134266,
        "Nopi_cos": 5.452564657261127,
        "Nopi_sinh": 7.955723540761046,
        "Nopi_abs": 6.333544134938385,
        "Nopi_+": 5.808163661224514,
        "Nopi_*": 5.002213595420244,
        "Nopi_fac": 10.0,
    }
    t = Tree(
        variables=["x%d" % i for i in range(5)],
        parameters=["a%d" % i for i in range(10)],
        x=xtrain,
        y=ytrain,
        prior_par=prior_par,
    )
    print(xtest)

    # Predict
    ypred = t.trace_predict(xtest, samples=samples, burnin=10000)

    print(ypred)
    print(ytest)
    print(50.0 * np.sin(xtest["x0"]) / xtest["x2"] - 4.0 * xtest["x1"] + 3)

    # Done
    return t


def test5(string="(P120 + (((ALPHACAT / _a2) + (_a2 * CDH3)) + _a0))"):
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
    for i in range(1000000):
        t.mcmc_step(verbose=True)
        print("-" * 150)
        t2 = Tree(from_string=str(t))
        print(t)
        print(t2)
        if str(t2) != str(t):
            raise

    return t


if __name__ == "__main__":
    NP, NS = 100, 1000
    test5()

if __name__ == "__main__":
    unittest.main()
