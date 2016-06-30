import sys
from copy import deepcopy
from random import seed, random, randint
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
                               variables=deepcopy(variables),
                               parameters=deepcopy(parameters),
                               prior_par=deepcopy(prior_par), x=x, y=y,
                               BT=1)}
        self.t1 = self.trees[1]
        for BT in [T for T in Ts if T != 1]:
            treetmp = Tree(ops=ops,
                           variables=deepcopy(variables),
                           parameters=deepcopy(parameters),
                           prior_par=deepcopy(prior_par), x=x, y=y,
                           root_value=str(self.t1),
                           BT=BT)
            self.trees[BT] = treetmp
            # Share fitted parameters and representative with other trees
            self.trees[BT].fit_par = self.t1.fit_par
            self.trees[BT].representative = self.t1.representative
        return

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """ Perform a MCMC step in each of the trees. """
        # Loop over all trees
        for T, tree in self.trees.items():
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
        # Done
        return

    # -------------------------------------------------------------------------
    def tree_swap(self):
        # Choose Ts to swap
        nT1 = randint(0, len(self.trees.keys())-2)
        nT2 = nT1 + 1
        Ts = self.trees.keys()
        Ts.sort()
        t1 = self.trees[Ts[nT1]]
        t2 = self.trees[Ts[nT2]]
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
            self.trees[Ts[nT1]] = t2
            self.trees[Ts[nT2]] = t1
            t1.BT = BT2
            t2.BT = BT1
            self.t1 = self.trees[1]
            return BT1, BT2
        # Done
        return None, None

    # -------------------------------------------------------------------------
    def anneal(self, n=10000, factor=5):
        # Heat up
        for t in self.trees.values():
            t.BT *= factor
        for kk in range(n):
            print >> sys.stderr, '# Annealing heating at %g: %d / %d' % (
                self.trees[1].BT, kk, n
            )
            self.mcmc_step()
            self.tree_swap()
        # Cool down
        for t in self.trees.values():
            t.BT /= factor
        for kk in range(2*n):
            print >> sys.stderr, '# Annealing cooling at %g: %d / %d' % (
                self.trees[1].BT, kk, 2*n
            )
            self.mcmc_step()
            self.tree_swap()
        # Done
        return
        

if __name__ == '__main__':
    sys.path.append('Validation/')
    import iodata
    sys.path.append('Prior')
    from fit_prior import read_prior_par
    from pprint import pprint

    # Temperatures
    Ts = [
        1,
        1.20,
        1.44,
        1.73,
        2.07,
        2.49,
        2.99,
        3.58,
        4.30,
        5.16,
    ]

    # Read the data
    prior_par = read_prior_par('Prior/prior_param_sq.named_equations.nv7.np7.2016-06-06 16:43:26.287530.dat')
    VARS = iodata.XVARS['Trepat']
    Y = iodata.YLABS['Trepat']
    inFileName = 'Validation/Trepat/data/%s' % (iodata.FNAMES['Trepat'])
    data, x, y = iodata.read_data(
        'Trepat', ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    #print x, y

    # Initialize the parallel object
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(7)],
        x=x, y=y,
        prior_par=prior_par,
    )

    NREP = 1000000
    for rep in range(NREP):
        print '=' * 77
        print rep, '/', NREP
        p.mcmc_step()
        print '>> Swaping:', p.tree_swap()
        pprint(p.trees)
        print '.' * 77
        for T in Ts:
            energy_ref = p.trees[T].get_energy(reset=False)[0]
            print T, '\t',  \
                p.trees[T].E, energy_ref, \
                p.trees[T].bic
            if abs(p.trees[T].E - energy_ref) > 1.e-6:
                print p.trees[T].canonical(), p.trees[T].representative[p.trees[T].canonical()]
                raise
            if p.trees[T].representative != p.trees[1].representative:
                pprint(p.trees[T].representative)
                pprint(p.trees[1].representative)
                raise
            if p.trees[T].fit_par != p.trees[1].fit_par:
                raise

