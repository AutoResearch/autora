"""This test verifies that, for a simple sampling process (i.e. with
small formulas, and few variables, parameters and operations, so that
the sampling is exhaustive) WITH data, the frequency with which a
certain formula is visited is proportional to its probability (that
is, that the sampler samples from the equilibrium distribution).

"""

import sys
import numpy as np

sys.path.append('..')
from mcmc import Tree

sys.path.append('../Validation')
from iodata import *


DEGCORRECT = True

data, x, y = read_data(
    'Trepat',
    in_fname = '../Validation/Trepat/data/cadhesome_protein.csv',
)


t = Tree(
    variables = list(x.columns.values),
    ops = {'+' : 2, '*' : 2},
    parameters=['a%d' % i for i in range(2)],
    x=x, y=y,
)
t.max_size = 5

count, energy, cannonical = {}, {}, {}
for rep in range(1000000):
    t.mcmc_step(p_rr=0.05, p_long=.45, degcorrect=DEGCORRECT)
    print >> sys.stderr, \
        rep, t, t.E, t.get_energy(degcorrect=DEGCORRECT), t.cannonical(), t.par_values
    if abs(t.E - t.get_energy(degcorrect=DEGCORRECT)) > 1.e-8:
        raise KKError
    try:
        count[str(t)] += 1
    except KeyError:
        count[str(t)] = 1
        energy[str(t)] = t.get_energy(degcorrect=DEGCORRECT)
        cannonical[str(t)] = t.cannonical()

for f in count:
    print ' '.join((
        '%g' % np.exp(-float(energy[f])),
        '%d' % count[f],
        str(f).replace(' ', ''),
        cannonical[f]
    ))
