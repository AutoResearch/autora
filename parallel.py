from copy import deepcopy
from random import random, randint
from numpy import exp
from mcmc import *

class Parallel():
    """ The Parallel class for parallel tempering. """

    # -------------------------------------------------------------------------
    def __init__(self, Ts, ops=OPS, variables=['x'], parameters=['a'],
                 prior_par={}, x=None, y=None):
        # All trees are initialized to the same tree but with different BT
        Ts.sort()
        self.trees = {1 : Tree(ops=ops,
                               variables=variables,
                               parameters=parameters,
                               prior_par=prior_par, x=x, y=y,
                               BT=1)}
        self.t1 = self.trees[1]
        for BT in [T for T in Ts if T != 1]:
            self.trees[BT] = deepcopy(self.t1)
            self.trees[BT].BT = BT
        # Common representative for all trees
        self.representative = {
            self.t1.canonical() : (
                str(self.t1),
                self.t1.E,
                self.t1.par_values,
            )
        }
        return

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """ Perform a MCMC step in each of the trees. """
        # Loop over all trees
        for tree in self.trees.values():
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
            # Update the representative of this canonical formula in all trees
            canonical = tree.canonical()
            if canonical not in self.representative:
                self.representative[canonical] = tree.representative[canonical]
                for t2 in self.trees.values():
                    t2.representative[canonical] = self.representative[canonical]
        # Done
        return

    # -------------------------------------------------------------------------
    def tree_swap(self):
        # Choose Ts to swap
        nT1 = randint(0, len(self.trees.keys())-2)
        nT2 = nT1 + 1
        t1 = self.trees[self.trees.keys()[nT1]]
        t2 = self.trees[self.trees.keys()[nT2]]
        # The temperatures and energies
        BT1, BT2 = t1.BT, t2.BT
        EB1, EB2, EP1, EP2 = t1.EB, t2.EB, t1.EP, t2.EP
        # The energy change
        DeltaE = EB1/BT2 + EB2/BT1 - (EB1/BT1 + EB2/BT2) 
        ## Option with log prior = entropy
        ##F1 = E1 - BT1*EP1 (change sign of last term if log prior = - entropy)
        ##F2 = E2 - BT2*EP2
        ##DeltaF = F1/BT2 + F2/BT1 - (F1/BT1 + F2/BT2) 
        # Accept change
        if random() < exp(-DeltaE):
            self.trees[BT1] = t2
            self.trees[BT2] = t1
            t1.BT = BT2
            t2.BT = BT1
            self.t1 = self.trees[1]
        # Done
        return


if __name__ == '__main__':
    sys.path.append('Validation/')
    import iodata
    sys.path.append('Prior')
    from fit_prior import read_prior_par
    from pprint import pprint

    # Temperatures
    Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Read the data
    prior_par = read_prior_par('Prior/prior_param_sq.named_equations.nv7.np7.2016-06-06 16:43:26.287530.dat')
    VARS = iodata.XVARS['Trepat']
    Y = iodata.YLABS['Trepat']
    inFileName = 'Validation/Trepat/data/%s' % (iodata.FNAMES['Trepat'])
    data, x, y = iodata.read_data(
        'Trepat', ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    print x, y

    # Initialize the parallel object
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(7)],
        x=x, y=y,
        prior_par=prior_par,
    )
    pprint(p.trees)

    for rep in range(100000):
        print '=' * 77
        p.mcmc_step()
        p.tree_swap()
        pprint(p.trees)
        print '.' * 77
        for T in p.trees:
            print T, p.trees[T].E, p.trees[T].get_energy(reset=False)[0]
            
