import numpy as np
import pandas as pd
import pytest  # noqa: 401

from autora.theorist.bms import Tree, get_priors


def test_tree_mcmc_stepping(
    num_points: int = 10,
    samples: int = 100,
    show_plot: bool = False,
    rng=np.random.default_rng(),
) -> Tree:
    """
    Testing the basic MCMC capacity. Note that even though an option (`show_plot`) is
    offered to compare the actual data (`y`) against the prediction, this test does not
    try to assess the prediction accuracy; it only ensures that the MCMC functionality
    can work bug-free.

    Parameters:
        num_points:
            the number of data points in each dimension of the synthetic data
        samples:
            the number of MCMC samples we want to get. The total MCMC iteration can be
            calculated as `burnin` + `samples`
        show_plot:
            whether to plot the predicted against actual response variable
        rng:
            initialization of random generator object

    Returns:
        the expression tree obtained from running the MCMC algorithm
    """

    # Create the data
    x = pd.DataFrame(
        dict([("x%d" % i, rng.uniform(0, 10, num_points)) for i in range(5)])
    )
    eps = rng.normal(0.0, 5, num_points)
    y = 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3 + eps

    # Create the formula
    t = Tree(
        variables=["x%d" % i for i in range(5)],
        parameters=["a%d" % i for i in range(10)],
        x=x,
        y=y,
        prior_par=get_priors()[0],
        BT=1.0,
    )

    # MCMC
    t.mcmc(burnin=200, thin=10, samples=samples, verbose=False)

    # Predict
    print(t.predict(x))
    print(y)
    print(50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)

    if show_plot:
        import matplotlib.pyplot as plt

        plt.plot(t.predict(x), 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)
        plt.show()

    return t
