"""This test verifies that, for a simple sampling process (i.e. with
small formulas, and few variables, parameters and operations, so that
the sampling is exhaustive) with no data, the frequency with which a
certain formula is visited is proportional to its probability (that
is, that the sampler samples from the equilibrium distribution).

"""

import sys
from numpy import exp
from pprint import pprint

sys.path.append('..')
from mcmc import Tree

t = Tree(
    ops={'+' : 2, 'sin': 1},
    prior_par={'Nopi_+' : 0, 'Nopi_sin' : 0},
)
"""
t = Tree(
    parameters=[],
    ops={'+' : 2},
    prior_par={'Nopi_+' : 0},
)
"""

t.max_size = 7

count, energy, canonical, representatives = {}, {}, {}, {}
links = {}
for rep in range(10000000):
    cano = t.canonical()
    if cano not in links:
        links[cano] = {}
    t.mcmc_step(p_rr=0.05, p_long=.45)
    print(rep+1, t, t.E, t.get_energy(), file=sys.stderr)
    if abs(t.E - t.get_energy()[0]) > 1.e-8:
        raise KKError
    can = t.canonical()
    try:
        links[cano][can] += 1
    except KeyError:
        links[cano][can] = 1
    try:
        count[str(t)] += 1
    except KeyError:
        count[str(t)] = 1
        energy[str(t)] = t.get_energy()
        canonical[str(t)] = can
    if can not in representatives:
        representatives[can] = {}
    try:
        representatives[can][str(t)] += 1
    except KeyError:
        representatives[can][str(t)] = 1

with open('test1_net.dat', 'w') as outf:
    for c1 in links:
        for c2 in links[c1]:
            print(c1.replace(' ', ''), c2.replace(' ', ''), links[c1][c2], file=outf)

with open('test1_out1.dat', 'w') as outf:
    for f in count:
        print(exp(-energy[f]), count[f], \
            canonical[f].replace(' ', ''), str(f).replace(' ', ''), file=outf)

with open('test1_out2.dat', 'w') as outf:
    for c in representatives:
        for s in representatives[c]:
            print(representatives[c][s], c, s, file=outf)
