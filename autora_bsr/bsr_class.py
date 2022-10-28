import copy
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from sklearn.base import BaseEstimator, RegressorMixin

from .funcs import Express, Node, allcal, display, genList, getNum, grow, newProp

_logger = logging.getLogger(__name__)


class BSRRegressor(BaseEstimator, RegressorMixin):
    """
    Bayesian Symbolic Regression
    """

    def __init__(
        self,
        treeNum=3,
        itrNum=5000,
        alpha1=0.4,
        alpha2=0.4,
        beta=-1,
        disp=False,
        val=100,
    ):
        self.treeNum = treeNum
        self.itrNum = itrNum
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.disp = disp
        self.val = val

    def model(self, last_ind=1):
        modd = []
        for i in range(self.treeNum):
            modd.append(Express(self.roots_[-last_ind][i]))
        return modd

    def complexity(self):
        compl = 0
        cmpls = []
        for i in range(self.treeNum):
            root_node = self.roots_[-1][i]
            numm = getNum(root_node)
            cmpls.append(numm)
            compl = compl + numm
        return compl

    def predict(self, test_data, method="last", last_ind=1):
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data)
        K = self.treeNum
        n_test = test_data.shape[0]
        XX = np.zeros((n_test, K))
        if method == "last":
            for countt in np.arange(K):
                temp = allcal(self.roots_[-last_ind][countt], test_data)
                temp.shape = temp.shape[0]
                XX[:, countt] = temp
            constant = np.ones((n_test, 1))
            XX = np.concatenate((constant, XX), axis=1)
            Beta = self.betas_[-last_ind]
            toutput = np.matmul(XX, Beta)
        return toutput

    # =============================================================================
    # # MCMC algorithm
    # K is the number of trees
    # MM is the number of iterations
    # alpha1, alpha2, beta are hyperparameters of priors
    # disp chooses whether to display intermediate results

    def fit(self, train_data, train_y):

        self.roots_ = []
        self.betas_ = []
        self.train_err_ = []

        # train_data must be a dataframe
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data)
        trainERRS = []
        ROOTS = []
        BETAS = []
        MM = self.itrNum
        K = self.treeNum
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        beta = self.beta

        if self.disp:
            _logger.info("Starting training.")

        while len(trainERRS) < MM:
            n_feature = train_data.shape[1]
            n_train = train_data.shape[0]
            """
            alpha1 = 0.4
            alpha2 = 0.4
            beta = -1
            """

            Ops = ["inv", "ln", "neg", "sin", "cos", "exp", "square", "cubic", "+", "*"]
            Op_weights = [1.0 / len(Ops)] * len(Ops)
            Op_type = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

            # List of tree samples
            RootLists = []
            for i in np.arange(K):
                RootLists.append([])

            SigaList = []  # List of sigma_a, for each component tree
            SigbList = []  # List of sigma_b, for each component tree

            sigma = invgamma.rvs(1)  # for output y

            # Initialization
            for count in np.arange(K):
                # create a new Root node
                Root = Node(0)
                sigma_a = invgamma.rvs(1)
                sigma_b = invgamma.rvs(1)

                # grow a tree from the Root node
                if self.disp:

                    _logger.info("Grow a tree from the Root node")

                grow(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                # Tree = genList(Root)

                # put the root into list
                RootLists[count].append(copy.deepcopy(Root))
                SigaList.append(sigma_a)
                SigbList.append(sigma_b)

            # calculate beta
            if self.disp:
                _logger.info("Calculate beta")
            # added a constant in the regression by fwl
            XX = np.zeros((n_train, K))
            for count in np.arange(K):
                temp = allcal(RootLists[count][-1], train_data)
                temp.shape = temp.shape[0]
                XX[:, count] = temp
            constant = np.ones((n_train, 1))  # added a constant
            XX = np.concatenate((constant, XX), axis=1)
            scale = np.max(np.abs(XX))
            XX = XX / scale
            epsilon = (
                np.eye(XX.shape[1]) * 1e-6
            )  # add to the matrix to prevent singular matrrix
            yy = np.array(train_y)
            yy.shape = (yy.shape[0], 1)
            Beta = np.linalg.inv(np.matmul(XX.transpose(), XX) + epsilon)
            Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
            output = np.matmul(XX, Beta)
            Beta = (
                Beta / scale
            )  # rescale the beta, above we scale XX for calculation by fwl

            total = 0
            accepted = 0
            errList = []
            totList = []
            nodeCounts = []

            tic = time.time()

            if self.disp:
                _logger.info("While total < ", self.val)
            while total < self.val:
                Roots = []  # list of current components
                # for count in np.arange(K):
                #     Roots.append(RootLists[count][-1])
                switch_label = False
                for count in np.arange(K):
                    Roots = []  # list of current components
                    for ccount in np.arange(K):
                        Roots.append(RootLists[ccount][-1])
                    # pick the root to be changed
                    sigma_a = SigaList[count]
                    sigma_b = SigbList[count]

                    # the returned Root is a new copy
                    if self.disp:
                        _logger.info("newProp...")
                    [res, sigma, Root, sigma_a, sigma_b] = newProp(
                        Roots,
                        count,
                        sigma,
                        train_y,
                        train_data,
                        n_feature,
                        Ops,
                        Op_weights,
                        Op_type,
                        beta,
                        sigma_a,
                        sigma_b,
                    )
                    if self.disp:
                        _logger.info("res:", res)
                        display(genList(Root))

                    total += 1
                    # update sigma_a and sigma_b
                    SigaList[count] = sigma_a
                    SigbList[count] = sigma_b

                    if res is True:
                        # flag = False
                        accepted += 1
                        # record newly accepted root
                        RootLists[count].append(copy.deepcopy(Root))

                        node_sums = 0
                        for k in np.arange(0, K):
                            node_sums += getNum(RootLists[k][-1])
                        nodeCounts.append(node_sums)

                        XX = np.zeros((n_train, K))
                        for i in np.arange(K):
                            temp = allcal(RootLists[i][-1], train_data)
                            temp.shape = temp.shape[0]
                            XX[:, i] = temp
                        constant = np.ones((n_train, 1))
                        XX = np.concatenate((constant, XX), axis=1)
                        scale = np.max(np.abs(XX))
                        XX = XX / scale
                        epsilon = (
                            np.eye(XX.shape[1]) * 1e-6
                        )  # add prevent singular matrix
                        yy = np.array(train_y)
                        yy.shape = (yy.shape[0], 1)
                        Beta = np.linalg.inv(np.matmul(XX.transpose(), XX) + epsilon)
                        Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))

                        output = np.matmul(XX, Beta)
                        Beta = (
                            Beta / scale
                        )  # rescale the beta, above we scale XX for calculation

                        error = 0
                        for i in np.arange(0, n_train):
                            error += (output[i, 0] - train_y[i]) * (
                                output[i, 0] - train_y[i]
                            )
                        rmse = np.sqrt(error / n_train)
                        errList.append(rmse)

                        if self.disp:

                            _logger.info(
                                "accept",
                                accepted,
                                "th after",
                                total,
                                "proposals and update ",
                                count,
                                "th component",
                            )
                            _logger.info(
                                "sigma:", round(sigma, 5), "error:", round(rmse, 5)
                            )  # ,"log.likelihood:",round(llh,5))

                            display(genList(Root))
                            _logger.info("---------------")
                        totList.append(total)
                        total = 0

                    # @fwl added condition to control running time
                    my_index = min(10, len(errList))
                    if (
                        len(errList) > 100
                        and 1
                        - np.min(errList[-my_index:]) / np.mean(errList[-my_index:])
                        < 0.05
                    ):
                        # converged
                        switch_label = True
                        break
                        # Roots[count] = oldRoot
                if switch_label:
                    break

            if self.disp:
                for i in np.arange(0, len(train_y)):
                    print(output[i, 0], train_y[i])

            toc = time.time()  # cauculate running time
            tictoc = toc - tic
            if self.disp:
                _logger.info("run time:{:.2f}s".format(tictoc))

                _logger.info("------")
                _logger.info("mean rmse of last 5 accepts:", np.mean(errList[-6:-1]))

            trainERRS.append(errList)
            ROOTS.append(Roots)
            BETAS.append(Beta)
        self.roots_ = ROOTS
        self.train_err_ = trainERRS
        self.betas_ = BETAS

        return


# =============================================================================
# # MCMC algorithm
# K is the number of trees
# MM is the number of iterations
# =============================================================================


def symreg(K, MM, train_data, test_data, train_y, test_y, disp=True):

    trainERRS = []
    testERRS = []
    ROOTS = []

    while len(trainERRS) < MM:
        n_feature = train_data.shape[1]
        n_train = train_data.shape[0]
        n_test = test_data.shape[0]

        beta = -1

        # Ops = ['inv', 'ln', 'neg', 'sin', 'cos', 'exp', '+', '*']
        # Op_weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        # Op_type = [1, 1, 1, 1, 1, 1, 2, 2]
        # n_op = len(Ops)

        Ops = ["inv", "ln", "neg", "sin", "cos", "exp", "square", "cubic", "+", "*"]
        Op_weights = [1.0 / len(Ops)] * len(Ops)
        Op_type = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

        # List of tree samples
        RootLists = []
        for i in np.arange(K):
            RootLists.append([])

        SigaList = []  # List of sigma_a, for each component tree
        SigbList = []  # List of sigma_b, for each component tree

        sigma = invgamma.rvs(1)  # for output y

        val = 100

        # Initialization
        for count in np.arange(K):
            # create a new Root node
            Root = Node(0)
            sigma_a = invgamma.rvs(1)
            sigma_b = invgamma.rvs(1)

            # grow a tree from the Root node
            grow(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

            # put the root into list
            RootLists[count].append(copy.deepcopy(Root))
            SigaList.append(sigma_a)
            SigbList.append(sigma_b)

        # calculate beta
        # added a constant in the regression by fwl
        XX = np.zeros((n_train, K))
        for count in np.arange(K):
            temp = allcal(RootLists[count][-1], train_data)
            temp.shape = temp.shape[0]
            XX[:, count] = temp
        constant = np.ones((n_train, 1))  # added a constant
        XX = np.concatenate((constant, XX), axis=1)
        scale = np.max(np.abs(XX))
        XX = XX / scale
        epsilon = (
            np.eye(XX.shape[1]) * 1e-6
        )  # add to the matrix to prevent singular matrix
        yy = np.array(train_y)
        yy.shape = (yy.shape[0], 1)
        Beta = np.linalg.inv(np.matmul(XX.transpose(), XX) + epsilon)
        Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
        output = np.matmul(XX, Beta)
        Beta = (
            Beta / scale
        )  # rescale the beta, above we scale XX for calculation by fwl

        total = 0
        accepted = 0
        errList = []
        totList = []
        testList = []
        nodeCounts = []

        tic = time.time()

        while total < val:
            Roots = []  # list of current components
            # for count in np.arange(K):
            #     Roots.append(RootLists[count][-1])
            switch_label = False
            for count in np.arange(K):
                Roots = []  # list of current components
                for ccount in np.arange(K):
                    Roots.append(RootLists[ccount][-1])
                # pick the root to be changed
                sigma_a = SigaList[count]
                sigma_b = SigbList[count]

                # the returned Root is a new copy
                [res, sigma, Root, sigma_a, sigma_b] = newProp(
                    Roots,
                    count,
                    sigma,
                    train_y,
                    train_data,
                    n_feature,
                    Ops,
                    Op_weights,
                    Op_type,
                    beta,
                    sigma_a,
                    sigma_b,
                )
                # print("res:",res)
                # display(genList(Root))

                total += 1
                # update sigma_a and sigma_b
                SigaList[count] = sigma_a
                SigbList[count] = sigma_b

                if res is True:
                    # flag = False
                    accepted += 1
                    # record newly accepted root
                    RootLists[count].append(copy.deepcopy(Root))

                    node_sums = 0
                    for k in np.arange(0, K):
                        node_sums += getNum(RootLists[k][-1])
                    nodeCounts.append(node_sums)

                    XX = np.zeros((n_train, K))
                    for i in np.arange(K):
                        temp = allcal(RootLists[i][-1], train_data)
                        temp.shape = temp.shape[0]
                        XX[:, i] = temp
                    constant = np.ones((n_train, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    scale = np.max(np.abs(XX))
                    XX = XX / scale
                    epsilon = (
                        np.eye(XX.shape[1]) * 1e-6
                    )  # add to prevent singular matrix
                    yy = np.array(train_y)
                    yy.shape = (yy.shape[0], 1)
                    Beta = np.linalg.inv(np.matmul(XX.transpose(), XX) + epsilon)
                    Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))

                    output = np.matmul(XX, Beta)
                    Beta = (
                        Beta / scale
                    )  # rescale the beta, above we scale XX for calculation

                    error = 0
                    for i in np.arange(0, n_train):
                        error += (output[i, 0] - train_y[i]) * (
                            output[i, 0] - train_y[i]
                        )
                    rmse = np.sqrt(error / n_train)
                    errList.append(rmse)

                    # compute test error
                    XX = np.zeros((n_test, K))
                    for countt in np.arange(K):
                        temp = allcal(RootLists[countt][-1], test_data)
                        temp.shape = temp.shape[0]
                        XX[:, countt] = temp
                    constant = np.ones((n_test, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    toutput = np.matmul(XX, Beta)

                    terror = 0
                    for i in np.arange(0, n_test):
                        terror += (toutput[i, 0] - test_y[i]) * (
                            toutput[i, 0] - test_y[i]
                        )
                    trmse = np.sqrt(terror / n_test)
                    testList.append(trmse)

                    if disp:
                        _logger.info(
                            "accept",
                            accepted,
                            "th after",
                            total,
                            "proposals and update ",
                            count,
                            "th component",
                        )
                        _logger.info(
                            "sigma:", round(sigma, 5), "error:", round(rmse, 5)
                        )  # ,"log.likelihood:",round(llh,5))

                        display(genList(Root))
                        _logger.info("---------------")
                    totList.append(total)
                    total = 0

                my_index = min(10, len(errList))
                if (
                    len(errList) > 100
                    and 1 - np.min(errList[-my_index:]) / np.mean(errList[-my_index:])
                    < 0.05
                ):
                    # converged
                    switch_label = True
                    break
            if switch_label:
                break

        for i in np.arange(0, len(train_y)):
            print(output[i, 0], train_y[i])

        toc = time.time()  # calculate running time
        tictoc = toc - tic
        if disp:
            _logger.info("run time:{:.2f}s".format(tictoc))

            _logger.info("------")
            _logger.info("mean rmse of last 5 accepts:", np.mean(errList[-6:-1]))
            _logger.info("mean rmse of last 5 tests:", np.mean(testList[-6:-1]))

        trainERRS.append(errList)
        testERRS.append(testList)
        ROOTS.append(Roots)

    return ROOTS
