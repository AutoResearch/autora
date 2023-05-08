import copy
import logging
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from autora.theorist.bsr.funcs import get_all_nodes, grow, prop_new
from autora.theorist.bsr.node import Node
from autora.theorist.bsr.prior import get_prior_dict

_logger = logging.getLogger(__name__)


class BSRRegressor(BaseEstimator, RegressorMixin):
    """
    Bayesian Symbolic Regression (BSR)

    A MCMC-sampling-based Bayesian approach to symbolic regression -- a machine learning method
    that bridges `X` and `y` by automatically building up mathematical expressions of basic
    functions. Performance and speed of `BSR` depends on pre-defined parameters.

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
        train_errs_: the training losses associated with the best-fit model
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
        prior_name: str = "Uniform",
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
        assert beta < 0, "beta must be < 0"
        assert tree_num > 0, "tree num must be > 0"
        assert alpha1 > 0 and alpha2 > 0, "alpha must > 0"
        assert itr_num > 0 and val > 0, "itr_num and val must > 0"

        self.tree_num = tree_num
        self.itr_num = itr_num
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.show_log = show_log
        self.val = val
        self.last_idx = last_idx
        self.prior_name = prior_name

        # attributes that are not set until `fit`
        self.roots_: Optional[List[List[Node]]] = None
        self.betas_: Optional[List[List[float]]] = None
        self.train_errs_: Optional[List[List[float]]] = None
        self.best_rmse: Optional[float] = None
        self.best_idx: int = 0

        self.X_: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.y_: Optional[Union[np.ndarray, pd.DataFrame]] = None

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

        assert self.roots_ and self.betas_
        for i in np.arange(k):
            tree_i = self.roots_[self.best_idx][i]
            tree_out = tree_i.evaluate(X)
            # tree_out.shape = tree_out.shape[0]
            tree_outs[:, i] = tree_out

        ones = np.ones((n_test, 1))
        tree_outs = np.concatenate((ones, tree_outs), axis=1)
        _beta = self.betas_[self.best_idx]
        output = np.matmul(tree_outs, _beta)

        return output

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ):
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
        train_errs: List[List[float]] = []
        roots: List[List[Node]] = []
        betas: List[List[float]] = []
        itr_num = self.itr_num
        k_tree = self.tree_num
        beta = self.beta

        best_idx, best_rmse_itr = 0, np.inf

        ops_name_lst, ops_weight_lst, ops_priors = get_prior_dict(
            prior_name=self.prior_name
        )

        n_train, n_feature = X.shape

        if self.show_log:
            _logger.info("Starting training...")

        while len(train_errs) < itr_num:
            # List of tree samples
            root_lists: List[List[Node]] = [[] for _ in range(k_tree)]

            sigma_a_list = []  # List of sigma_a, for each component tree
            sigma_b_list = []  # List of sigma_b, for each component tree

            sigma_y = invgamma.rvs(1)  # for output y

            # Initialization: make sure the initial tree must be valid
            is_init_valid = False

            while not is_init_valid:
                for count in np.arange(k_tree):
                    # create a new root node
                    root = Node(0)
                    sigma_a = invgamma.rvs(1)
                    sigma_b = invgamma.rvs(1)

                    grow(
                        root,
                        ops_name_lst,
                        ops_weight_lst,
                        ops_priors,
                        n_feature,
                        sigma_a=sigma_a,
                        sigma_b=sigma_b,
                    )

                    # put the root into list
                    root_lists[count].append(root)
                    sigma_a_list.append(sigma_a)
                    sigma_b_list.append(sigma_b)

                # added a constant in the regression by fwl
                tree_outputs = np.zeros((n_train, k_tree))

                for count in np.arange(k_tree):
                    tree_outputs[:, count] = root_lists[count][-1].evaluate(X)

                tree_outputs = np.concatenate((np.ones((n_train, 1)), tree_outputs), axis=1)

                # handle potential invalid tree generated, e.g. log on negative input
                if np.isfinite(tree_outputs).all():
                    scale = np.max(np.abs(tree_outputs))
                    tree_outputs = tree_outputs / scale
                    epsilon = (
                        np.eye(tree_outputs.shape[1]) * 1e-6
                    )  # add to the matrix to prevent singular matrrix
                    ln_beta = np.linalg.inv(
                        np.matmul(tree_outputs.transpose(), tree_outputs) + epsilon
                    )
                    ln_beta = np.matmul(
                        ln_beta, np.matmul(tree_outputs.transpose(), y)
                    )

                    output = np.matmul(tree_outputs, ln_beta)
                    # rescale the beta, above we scale tree_outputs for calculation
                    ln_beta /= scale
                    best_rmse = np.sqrt(np.sum((output - y) ** 2) / n_train)
                    best_ln_beta = ln_beta
                    is_init_valid = True

            total = 0
            accepted = 0
            errs = []

            while total < self.val:
                switch_label = False
                for count in range(k_tree):
                    # list of current components
                    curr_roots = [root_lists[i][-1] for i in np.arange(k_tree)]

                    # pick the root to be changed
                    sigma_a = sigma_a_list[count]
                    sigma_b = sigma_b_list[count]
                    # the returned root is a new copy
                    res, root, sigma_y, sigma_a, sigma_b = prop_new(
                        curr_roots,
                        count,
                        sigma_y,
                        beta,
                        sigma_a,
                        sigma_b,
                        X,
                        y,
                        ops_name_lst,
                        ops_weight_lst,
                        ops_priors,
                    )

                    total += 1
                    # update sigma_a and sigma_b
                    sigma_a_list[count] = sigma_a
                    sigma_b_list[count] = sigma_b
                    if res:
                        accepted += 1

                        tree_outputs = np.zeros((n_train, k_tree))

                        for i in np.arange(k_tree):
                            tree_outputs[:, i] = root_lists[i][-1].evaluate(X)

                        # if an accepted tree is invalid, do not consider it
                        if not np.isfinite(tree_outputs).all():
                            continue

                        tree_outputs = np.concatenate((np.ones((n_train, 1)), tree_outputs), axis=1)
                        scale = np.max(np.abs(tree_outputs))
                        tree_outputs = tree_outputs / scale
                        epsilon = (
                            np.eye(tree_outputs.shape[1]) * 1e-6
                        )  # add to prevent singular matrix

                        ln_beta = np.linalg.inv(
                            np.matmul(tree_outputs.transpose(), tree_outputs) + epsilon
                        )
                        ln_beta = np.matmul(
                            ln_beta, np.matmul(tree_outputs.transpose(), y)
                        )

                        # output = np.matmul(tree_outputs, ln_beta)[:, 0]
                        output = np.matmul(tree_outputs, ln_beta)
                        # rescale the beta, above we scale tree_outputs for calculation
                        ln_beta /= scale

                        rmse = np.sqrt(np.sum((output - y) ** 2) / n_train)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_ln_beta = ln_beta
                            # record newly accepted root
                            root_lists[count].append(root)
                        errs.append(rmse)

                    if len(errs) > 100:
                        lapses = min(10, len(errs))
                        converge_ratio = 1 - np.min(errs[-lapses:]) / np.mean(
                            errs[-lapses:]
                        )
                        if converge_ratio < 0.05:
                            # converged
                            switch_label = True
                            break
                if switch_label:
                    break

            if best_rmse < best_rmse_itr:
                best_rmse_itr = best_rmse
                best_idx = len(roots)

            if self.show_log:
                _logger.info(f"iteration {len(roots)}, best rmse: {best_rmse_itr}")

            train_errs.append(errs)
            roots.append([root_lists[i][-1] for i in np.arange(k_tree)])
            betas.append(best_ln_beta)

        if self.show_log:
            _logger.info(f"Training finished, best {k_tree} trees are:")
            for i in np.arange(k_tree):
                _logger.info(roots[best_idx][i])
            _logger.info(f"best beta is: {betas[best_idx]}")

        self.roots_ = roots
        self.train_errs_ = train_errs
        self.betas_ = betas
        self.X_, self.y_ = X, y
        self.best_rmse, self.best_idx = best_rmse_itr, best_idx
        return self

    def _model(self, last_ind: int = 1) -> List[str]:
        """
        Return the models in the last-i-th iteration, default `last_ind = 1` refers to the
        last (final) iteration.
        """
        models = []
        assert self.roots_
        for i in range(self.tree_num):
            models.append(self.roots_[-last_ind][i].get_expression())
        return models

    def _complexity(self) -> int:
        """
        Return the complexity of the final models, which equals to the sum of nodes in all
        expression trees.
        """
        cp = 0
        assert self.roots_
        for i in range(self.tree_num):
            root_node = self.roots_[-1][i]
            num = len(get_all_nodes(root_node))
            cp = cp + num
        return cp
