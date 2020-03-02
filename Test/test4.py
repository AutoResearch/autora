"""This test verifies that, for a simple sampling process (i.e. with
small formulas, and few variables, parameters and operations, so that
the sampling is exhaustive) WITH data, the frequency with which a
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
t.x.to_csv('test4_datax.csv', index=False)
t.y.to_csv('test4_datay.csv', index=False)

for rep in range(10000):
    print('bi%d' % rep, t, file=sys.stderr)
    t.mcmc_step()


count, energy, canonical, representatives = {}, {}, {}, {}
links = {}
for rep in range(10000000):
    cano = t.canonical()
    if cano not in links:
        links[cano] = {}
    t.mcmc_step()
    print(rep+1, t, t.E, t.get_energy(), file=sys.stderr)
    if abs(t.E - t.get_energy()) > 1.e-8:
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

with open('test4_net.dat', 'w') as outf:
    for c1 in links:
        for c2 in links[c1]:
            print(c1.replace(' ', ''), c2.replace(' ', ''), links[c1][c2], file=outf)

with open('test4_out1.dat', 'w') as outf:
    for f in count:
        print(np.exp(-float(energy[f])), count[f], \
            canonical[f].replace(' ', ''), str(f).replace(' ', ''), file=outf)

with open('test4_out2.dat', 'w') as outf:
    for c in representatives:
        for s in representatives[c]:
            print(representatives[c][s], c, s, file=outf)
