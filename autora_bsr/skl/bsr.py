import copy
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from autora_bsr.funcs import (
    Express,
    Node,
    allcal,
    display,
    genList,
    getNum,
    grow,
    newProp,
)

from typing import Union, Optional, List

_logger = logging.getLogger(__name__)


class BSRRegressor(BaseEstimator, RegressorMixin):
    """
    Bayesian Symbolic Regression (BSR)

    A MCMC-sampling-based Bayesian approach to symbolic regression -- a machine learning method
    that bridges `X` and `y` by automatically building up mathematical expressions of basic functions.
    Performance and speed of `BSR` depends on pre-defined parameters.

    This class is intended to be compatible with the
    [Scikit-Learn Estimator API](https://scikit-learn.org/stable/developers/develop.html).

    Examples:

        >>> import numpy as np
        >>> num_samples = 1000
        >>> X = np.linspace(start=0, stop=1, num=num_samples).reshape(-1, 1)
        >>> y = np.sqrt(X)
        >>> estimator = BSRRegressor()
        >>> estimator = estimator.fit(X, y)
        >>> estimator.predict([[1.5]])

    Attributes:
        roots_: the root(s) of the best-fit symbolic regression (SR) tree(s)
        betas_: the beta parameters of the best-fit model
        train_errs: the training losses associated with the best-fit model
    """

    def __init__(
        self,
        tree_num: int = 3,
        itr_num: int = 5000,
        alpha1: float = 0.4,
        alpha2: float = 0.4,
        beta: float = -1,
        show_log: bool = False,
        val: int = 100,
        last_idx: int = -1,
    ):
        """
        Arguments:
            tree_num: pre-specified number of SR trees to fit in the model
            itr_num: number of iterations steps to run for the model fitting process
            alpha1, alpha2, beta: the hyper-parameters of priors
            show_log: whether to output certain logging info
            val: number of validation steps to run for each iteration step
            last_idx: the index of which latest (most best-fit) model to use
                (-1 means the latest one)
        """
        self.tree_num = tree_num
        self.itr_num = itr_num
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.show_log = show_log
        self.val = val
        self.last_idx = last_idx

        self.roots_ = []
        self.betas_ = []
        self.train_errs_ = []

        self.X_: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.y_: Optional[Union[np.ndarray, pd.DataFrame]] = None

    def model(self, last_ind: int = 1) -> List[str]:
        models = []
        for i in range(self.tree_num):
            models.append(Express(self.roots_[-last_ind][i]))
        return models

    def complexity(self) -> int:
        cp = 0
        cps = []
        for i in range(self.tree_num):
            root_node = self.roots_[-1][i]
            num = getNum(root_node)
            cps.append(num)
            cp = cp + num
        return cp

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Applies the fitted model to a set of independent variables `X`,
        to give predictions for the dependent variable `y`.

        Arguments:
            X: independent variables in an n-dimensional array
        Returns:
            y: predicted dependent variable values
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        check_is_fitted(self, attributes=["roots_"])

        k = self.tree_num
        n_test = X.shape[0]
        tree_outs = np.zeros((n_test, k))

        for i in np.arange(k):
            tree_out = allcal(self.roots_[-self.last_idx][i], X)
            tree_out.shape = tree_out.shape[0]
            tree_outs[:, i] = tree_out

        ones = np.ones((n_test, 1))
        tree_outs = np.concatenate((ones, tree_outs), axis=1)
        beta = self.betas_[-self.last_idx]
        output = np.matmul(tree_outs, beta)

        return output

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]):
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array
        Returns:
            self (BSR): the fitted estimator
        """
        # train_data must be a dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        train_errs = []
        roots = []
        betas = []
        itr_num = self.itr_num
        k = self.tree_num
        beta = self.beta

        if self.show_log:
            _logger.info("Starting training")
        while len(train_errs) < itr_num:
            n_feature = X.shape[1]
            n_train = X.shape[0]

            ops = ["inv", "ln", "neg", "sin", "cos", "exp", "square", "cubic", "+", "*"]
            op_weights = [1.0 / len(ops)] * len(ops)
            op_type = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

            # List of tree samples
            root_lists = [[] for _ in range(k)]

            sigma_a_list = []  # List of sigma_a, for each component tree
            sigma_b_list = []  # List of sigma_b, for each component tree

            sigma = invgamma.rvs(1)  # for output y

            # Initialization
            for count in np.arange(k):
                # create a new root node
                root = Node(0)
                sigma_a = invgamma.rvs(1)
                sigma_b = invgamma.rvs(1)

                # grow a tree from the root node
                if self.show_log:
                    _logger.info("Grow a tree from the root node")
                grow(root, n_feature, ops, op_weights, op_type, beta, sigma_a, sigma_b)
                # Tree = genList(root)

                # put the root into list
                root_lists[count].append(copy.deepcopy(root))
                sigma_a_list.append(sigma_a)
                sigma_b_list.append(sigma_b)

            # calculate beta
            if self.show_log:
                _logger.info("Calculate beta")
            # added a constant in the regression by fwl
            tree_outputs = np.zeros((n_train, k))

            for count in np.arange(k):
                temp = allcal(root_lists[count][-1], X)
                temp.shape = temp.shape[0]
                tree_outputs[:, count] = temp

            constant = np.ones((n_train, 1))  # added a constant
            tree_outputs = np.concatenate((constant, tree_outputs), axis=1)
            scale = np.max(np.abs(tree_outputs))
            tree_outputs = tree_outputs / scale
            epsilon = (
                np.eye(tree_outputs.shape[1]) * 1e-6
            )  # add to the matrix to prevent singular matrrix
            yy = np.array(y)
            yy.shape = (yy.shape[0], 1)
            beta = np.linalg.inv(np.matmul(tree_outputs.transpose(), tree_outputs) + epsilon)
            beta = np.matmul(beta, np.matmul(tree_outputs.transpose(), yy))
            output = np.matmul(tree_outputs, beta)
            # rescale the beta, above we scale tree_outputs for calculation by fwl
            beta /= scale

            total = 0
            accepted = 0
            errs = []
            total_list = []
            node_counts = []

            tic = time.time()

            if self.show_log:
                _logger.info("While total < ", self.val)
            while total < self.val:
                curr_roots = []  # list of current components

                switch_label = False
                for count in np.arange(k):
                    curr_roots = []  # list of current components
                    for i in np.arange(k):
                        curr_roots.append(root_lists[i][-1])
                    # pick the root to be changed
                    sigma_a = sigma_a_list[count]
                    sigma_b = sigma_b_list[count]

                    # the returned root is a new copy
                    if self.show_log:
                        _logger.info("newProp...")
                    res, sigma, root, sigma_a, sigma_b = newProp(
                        curr_roots,
                        count,
                        sigma,
                        y,
                        X,
                        n_feature,
                        ops,
                        op_weights,
                        op_type,
                        beta,
                        sigma_a,
                        sigma_b,
                    )
                    if self.show_log:
                        _logger.info("res:", res)
                        display(genList(root))

                    total += 1
                    # update sigma_a and sigma_b
                    sigma_a_list[count] = sigma_a
                    sigma_b_list[count] = sigma_b

                    if res is True:
                        # flag = False
                        accepted += 1
                        # record newly accepted root
                        root_lists[count].append(copy.deepcopy(root))

                        node_sums = 0
                        for k in np.arange(0, k):
                            node_sums += getNum(root_lists[k][-1])
                        node_counts.append(node_sums)

                        tree_outputs = np.zeros((n_train, k))

                        for i in np.arange(k):
                            temp = allcal(root_lists[count][-1], X)
                            temp.shape = temp.shape[0]
                            tree_outputs[:, i] = temp

                        constant = np.ones((n_train, 1))
                        tree_outputs = np.concatenate((constant, tree_outputs), axis=1)
                        scale = np.max(np.abs(tree_outputs))
                        tree_outputs = tree_outputs / scale
                        epsilon = (
                            np.eye(tree_outputs.shape[1]) * 1e-6
                        )  # add to prevent singular matrix
                        yy = np.array(y)
                        yy.shape = (yy.shape[0], 1)
                        beta = np.linalg.inv(np.matmul(tree_outputs.transpose(), tree_outputs) + epsilon)
                        beta = np.matmul(beta, np.matmul(tree_outputs.transpose(), yy))

                        output = np.matmul(tree_outputs, beta)
                        beta = (
                            beta / scale
                        )  # rescale the beta, above we scale tree_outputs for calculation

                        error = 0
                        for i in np.arange(0, n_train):
                            error += (output[i, 0] - y[i]) * (
                                    output[i, 0] - y[i]
                            )

                        rmse = np.sqrt(error / n_train)
                        errs.append(rmse)

                        total_list.append(total)
                        total = 0

                    lapses = min(10, len(errs))
                    converge_ratio = 1 - np.min(errs[-lapses:]) / np.mean(errs[-lapses:])
                    if lapses > 100 and converge_ratio < 0.05:
                        # converged
                        switch_label = True
                        break
                if switch_label:
                    break

            if self.show_log:
                for i in np.arange(0, len(y)):
                    _logger.info(output[i, 0], y[i])
            toc = time.time()
            tictoc = toc - tic
            if self.show_log:
                _logger.info("Run time: {:.2f}s".format(tictoc))

                _logger.info("------")
                _logger.info("Mean rmse of last 5 accepts: {}".format(np.mean(errs[-6:-1])))

            train_errs.append(errs)
            roots.append(curr_roots)
            betas.append(beta)

        self.roots_ = roots
        self.train_errs_ = train_errs
        self.betas_ = betas
        self.X_, self.y_ = X, y
        return self
