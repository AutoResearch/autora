"""This test verifies that, for a simple sampling process (i.e. with
small formulas, and few variables, parameters and operations, so that
the sampling is exhaustive) with no data, the frequency with which a
certain formula is visited is proportional to its probability (that
is, that the sampler samples from the equilibrium distribution).

"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint


sys.path.append('..')
from mcmc import Tree

DEGCORRECT = True

def generate_data():
    x = pd.DataFrame({'x' : np.linspace(0, 10, 50)})
    y = 2.5 + x['x'] + np.sin(x['x']) + np.random.normal(0, 2., 50)
    return x, y


x, y = generate_data()
plt.plot(x['x'], y)
plt.plot(x['x'], 2.5 + x['x'] + np.sin(x['x']))
plt.plot(x['x'], 2.5 + x['x'] + np.sin(np.sin(x['x'])))
plt.plot(x['x'], 2.5 + x['x'])
plt.savefig('test4_data.pdf')
plt.show()


t = Tree(
    x=x, y=y,
    ops={'+' : 2, 'sin': 1},
    prior_par={'Nopi_+' : 0, 'Nopi_sin' : 0},
)
t.max_size = 7
print t.x
print t.y

for rep in range(10000):
    print >> sys.stderr, 'bi%d' % rep, t
    t.mcmc_step(p_rr=0.05, p_long=.45, degcorrect=DEGCORRECT)


count, energy, cannonical, representatives = {}, {}, {}, {}
links = {}
for rep in range(10000000):
    cano = t.cannonical()
    if cano not in links:
        links[cano] = {}
    t.mcmc_step(p_rr=0.05, p_long=.45, degcorrect=DEGCORRECT)
    print >> sys.stderr, \
        rep+1, t, t.E, t.get_energy(degcorrect=DEGCORRECT), t.n_commute
    if abs(t.E - t.get_energy(degcorrect=DEGCORRECT)) > 1.e-8:
        raise KKError
    can = t.cannonical()
    try:
        links[cano][can] += 1
    except KeyError:
        links[cano][can] = 1
    try:
        count[str(t)] += 1
    except KeyError:
        count[str(t)] = 1
        energy[str(t)] = t.get_energy(degcorrect=DEGCORRECT)
        cannonical[str(t)] = can
    if can not in representatives:
        representatives[can] = {}
    try:
        representatives[can][str(t)] += 1
    except KeyError:
        representatives[can][str(t)] = 1

with open('test4_net.tmp', 'w') as outf:
    for c1 in links:
        for c2 in links[c1]:
            print >> outf, \
                c1.replace(' ', ''), c2.replace(' ', ''), links[c1][c2]

with open('test4_out1.tmp', 'w') as outf:
    for f in count:
        print >> outf, np.exp(-float(energy[f])), count[f], \
            cannonical[f].replace(' ', ''), str(f).replace(' ', '')

with open('test4_out2.tmp', 'w') as outf:
    for c in representatives:
        for s in representatives[c]:
            print >> outf, representatives[c][s], c, s
