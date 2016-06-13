from random import randint
from mcmc import *

class Parallel():
    """ The Parallel class for parallel tempering. """

    # -------------------------------------------------------------------------
    def __init__(self, Ts, ops=OPS, variables=['x'], parameters=['a'],
                 prior_par={}, x=None, y=None):
        # All trees are initialized to the same tree but with different BT
        Ts.sort()
        self.trees = dict([(T, Tree(ops=ops, variables=variables,
                                    parameters=parameters,
                                    prior_par=prior_par, x=x, y=y,
                                    from_string(variables[0]), BT=T))
                           for T in Ts]) 
        # Common representative for all trees
        self.representative = {
            self.trees[0].canonical() : (
                str(self.trees[0]),
                self.trees[0].E,
                self.trees[0].par_values
            )
        }
       return

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """ Perform a MCMC step in each of the trees. """
        # Loop over all trees
        for tree in self.trees.values:
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
            # Update the representative of this canonical formula in all trees
            canonical = tree.canonical()
            if canonical not in self.representative:
                self.representative[canonical] = tree.representative[canonical]
                for t in self.trees:
                    t.representative[canonical] = self.representative[canonical]
        # Done
        return

    # -------------------------------------------------------------------------
    def tree_swap(self):
        # Choose Ts to swap
        nT1 = randint(0, len(self.trees.keys())-2)
        nT2 = nT1 + 1
        t1 = self.trees[self.trees.keys()[nT1]]
        t2 = self.trees[self.trees.keys()[nT2]]
        pass


if __name__ == '__main__':
    
