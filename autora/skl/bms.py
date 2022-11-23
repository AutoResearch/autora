from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from autora.theorist.bms import Parallel, Tree, get_priors, utils

_logger = logging.getLogger(__name__)

# hyperparameters for BMS
# 1) Priors for MCMC
PRIORS, _ = get_priors()

# 2) Temperatures for parallel tempering
TEMPERATURES = [1.0] + [1.04**k for k in range(1, 20)]


class BMSRegressor(BaseEstimator, RegressorMixin):
    """
    Bayesian Machine Scientist.

    BMS finds an optimal function to explain a dataset, given a set of variables,
    and a pre-defined number of parameters

    This class is intended to be compatible with the
    [Scikit-Learn Estimator API](https://scikit-learn.org/stable/developers/develop.html).

    Examples:

        >>> from autora.theorist.bms import Parallel, utils
        >>> import numpy as np
        >>> num_samples = 1000
        >>> X = np.linspace(start=0, stop=1, num=num_samples).reshape(-1, 1)
        >>> y = 15. * np.ones(num_samples)
        >>> estimator = BMSRegressor()
        >>> estimator = estimator.fit(X, y)
        >>> estimator.predict([[15.]])
        array([[15.]])


    Attributes:
        pms: the bayesian (parallel) machine scientist model
        model_: represents the best-fit model
        loss_: represents loss associated with best-fit model
        cache_: record of loss_ over model fitting epochs
    """

    def __init__(
        self,
        prior_par: dict = PRIORS,
        ts: List[float] = TEMPERATURES,
        epochs: int = 1500,
    ):
        """
        Arguments:
            prior_par: a dictionary of the prior probabilities of different functions based on
                wikipedia data scraping
            ts: contains a list of the temperatures that the parallel ms works at
        """
        self.ts = ts
        self.prior_par = prior_par
        self.epochs = epochs
        self.pms: Parallel = Parallel(Ts=ts)

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.model_: Tree = Tree()
        self.loss_: float = np.inf
        self.cache_: List = []
        self.variables: List = []

    def fit(self, X: np.ndarray, y: np.ndarray, num_param: int = 1) -> BMSRegressor:
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array
            num_param: number of parameters

        Returns:
            self (BMS): the fitted estimator
        """
        # firstly, store the column names of X since checking will
        # cast the type of X to np.ndarray
        if hasattr(X, "columns"):
            self.variables = list(X.columns)
        else:
            # create variables X_1 to X_n where n is the number of columns in X
            self.variables = ["X%d" % i for i in range(X.shape[1])]

        X, y = check_X_y(X, y)

        # cast X into pd.Pandas again to fit the need in mcmc.py
        X = pd.DataFrame(X, columns=self.variables)
        y = pd.Series(y)
        _logger.info("BMS fitting started")

        self.pms = Parallel(
            Ts=self.ts,
            variables=self.variables,
            parameters=["a%d" % i for i in range(num_param)],
            x=X,
            y=y,
            prior_par=self.prior_par,
        )
        self.model_, self.loss_, self.cache_ = utils.run(self.pms, self.epochs)

        _logger.info("BMS fitting finished")
        self.X_, self.y_ = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted model to a set of independent variables `X`,
        to give predictions for the dependent variable `y`.

        Arguments:
            X: independent variables in an n-dimensional array

        Returns:
            y: predicted dependent variable values
        """
        # this validation step will cast X into np.ndarray format
        X = check_array(X)

        check_is_fitted(self, attributes=["model_"])

        assert self.model_ is not None
        # we need to cast it back into pd.DataFrame with the original
        # column names (generated in `fit`).
        # in the future, we might need to look into mcmc.py to remove
        # these redundant type castings.
        X = pd.DataFrame(X, columns=self.variables)

        return np.expand_dims(self.model_.predict(X).to_numpy(), axis=1)

    def present_results(self):
        """
        Prints out the best equation, its description length,
        along with a plot of how this has progressed over the course of the search tasks.
        """
        check_is_fitted(self, attributes=["model_", "loss_", "cache_"])
        assert self.model_ is not None
        assert self.loss_ is not None
        assert self.cache_ is not None

        utils.present_results(self.model_, self.loss_, self.cache_)
