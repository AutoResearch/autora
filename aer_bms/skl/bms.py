from typing import List

import numpy as np
import pandas as pd
import sklearn.utils

from aer_bms import Parallel, utils
from aer_bms.mcmc import Tree

priors = {
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


temperatures = [1.0] + [1.04**k for k in range(1, 20)]


class BMS:
    """
    Bayesian Machine Scientist.

    BMS finds an optimal function to explain a dataset, given a set of variables,
    and a pre-defined number of parameters

    This class is intended to be compatible with the
    [Scikit-Learn Estimator API](https://scikit-learn.org/stable/developers/develop.html).

    Examples:

        >>> from aer_bms import Parallel, utils
        >>> import numpy as np
        >>> num_samples = 1000
        >>> X = np.linspace(start=0, stop=1, num=num_samples).reshape(-1, 1)
        >>> y = 15. * np.ones(num_samples)
        >>> estimator = BMS()
        >>> estimator = estimator.fit(X, y)
        >>> estimator.predict([[15.]])
        "place holder --- PLEASE FIX ---"


    Attributes:
        pms: the bayesian (parallel) machine scientist model
        model_: represents the best-fit model
    """

    def __init__(
        self,
        prior_par: dict = priors,
        ts: List[float] = temperatures,
        epochs: int = 3000,
    ) -> None:
        """
        Arguments:
            prior_par: a dictionary of the prior likelihoods of different functions based on
            wikipedia data scraping
            ts: contains a list of the temperatures that the parallel ms works at
        """
        self.ts = ts
        self.prior_par = prior_par
        self.epochs = epochs
        self.pms: Parallel = Parallel(Ts=[])
        self.model_: Tree = Tree()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array

        Returns:
            self (BMS): the fitted estimator
        """
        X, y = sklearn.utils.check_X_y(X, y)
        self.pms = Parallel(
            Ts=self.ts,
            variables=pd.DataFrame(range(len(X))),
            parameters=["a%d" % i for i in range(len(X))],
            x=X,
            y=y,
            prior_par=self.prior_par,
        )
        model, model_len, desc_len = utils.run(self.pms, self.epochs)
        self.model_ = model
        return self

    def predict(self, X):
        """
        Applies the fitted model to a set of independent variables `X`,
        to give predictions for the dependent variable `y`.

        Arguments:
            X: independent variables in an n-dimensional array

        Returns:
            y: predicted dependent variable values
        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.check_array(X)
        print(type(X))
        if type(X) == np.ndarray:
            x = pd.DataFrame(X)
        elif type(X) == pd.DataFrame:
            x = X
        else:
            raise TypeError("x must be either a dict or a pandas.DataFrame")
        return self.model_.predict(x)
