import sys
from copy import deepcopy
from random import randint, random
from typing import Optional, Tuple

import numpy as np
from numpy import exp

from .mcmc import Tree
from .prior import get_priors


class Parallel:
    """
    The Parallel Machine Scientist Object, equipped with parallel tempering

    Attributes:
        Ts: list of parallel temperatures
        trees: list of parallel trees, corresponding to each parallel temperature
        t1: equation tree which best describes the data
    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        Ts: list,
        ops=get_priors()[1],
        variables=["x"],
        parameters=["a"],
        max_size=50,
        prior_par=get_priors()[0],
        x=None,
        y=None,
    ) -> None:
        """
        Initialises Parallel Machine Scientist

        Args:
            Ts: list of temperature values
            ops: allowed operations for the search task
            variables: independent variables from data
            parameters: settable values to improve model fit
            max_size: maximum size (number of nodes) in a tree
            prior_par: prior values over ops
            x: independent variables of dataset
            y: dependent variable of dataset
        """
        # All trees are initialized to the same tree but with different BT
        Ts.sort()
        self.Ts = [str(T) for T in Ts]
        self.trees = {
            "1": Tree(
                ops=ops,
                variables=deepcopy(variables),
                parameters=deepcopy(parameters),
                prior_par=deepcopy(prior_par),
                x=x,
                y=y,
                max_size=max_size,
                BT=1,
            )
        }
        self.t1 = self.trees["1"]
        for BT in [T for T in self.Ts if T != 1]:
            treetmp = Tree(
                ops=ops,
                variables=deepcopy(variables),
                parameters=deepcopy(parameters),
                prior_par=deepcopy(prior_par),
                x=x,
                y=y,
                root_value=str(self.t1),
                max_size=max_size,
                BT=float(BT),
            )
            self.trees[BT] = treetmp
            # Share fitted parameters and representative with other trees
            self.trees[BT].fit_par = self.t1.fit_par
            self.trees[BT].representative = self.t1.representative

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=0.45) -> None:
        """
        Perform a MCMC step in each of the trees
        """
        # Loop over all trees
        for T, tree in list(self.trees.items()):
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
        self.t1 = self.trees["1"]

    # -------------------------------------------------------------------------
    def tree_swap(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Choose a pair of trees of adjacent temperatures and attempt to swap their temperatures
        based on the resultant energy change

        Returns: new temperature values for the pair of trees
        """
        # Choose Ts to swap
        nT1 = randint(0, len(self.Ts) - 2)
        nT2 = nT1 + 1
        t1 = self.trees[self.Ts[nT1]]
        t2 = self.trees[self.Ts[nT2]]
        # The temperatures and energies
        BT1, BT2 = t1.BT, t2.BT
        EB1, EB2 = t1.EB, t2.EB
        # The energy change
        DeltaE = np.float(EB1) * (1.0 / BT2 - 1.0 / BT1) + np.float(EB2) * (
            1.0 / BT1 - 1.0 / BT2
        )
        if DeltaE > 0:
            paccept = exp(-DeltaE)
        else:
            paccept = 1.0
        # Accept/reject change
        if random() < paccept:
            self.trees[self.Ts[nT1]] = t2
            self.trees[self.Ts[nT2]] = t1
            t1.BT = BT2
            t2.BT = BT1
            self.t1 = self.trees["1"]
            return self.Ts[nT1], self.Ts[nT2]
        else:
            return None, None

    # -------------------------------------------------------------------------
    def anneal(self, n=1000, factor=5) -> None:
        """
        Annealing function for the Machine Scientist

        Args:
            n: number of mcmc step & tree swap iterations
            factor: degree of annealing - how much the temperatures are raised

        Returns: Nothing

        """
        for t in list(self.trees.values()):
            t.BT *= factor
        for kk in range(n):
            print(
                "# Annealing heating at %g: %d / %d" % (self.trees["1"].BT, kk, n),
                file=sys.stderr,
            )
            self.mcmc_step()
            self.tree_swap()
        # Cool down (return to original temperatures)
        for BT, t in list(self.trees.items()):
            t.BT = float(BT)
        for kk in range(2 * n):
            print(
                "# Annealing cooling at %g: %d / %d" % (self.trees["1"].BT, kk, 2 * n),
                file=sys.stderr,
            )
            self.mcmc_step()
            self.tree_swap()
