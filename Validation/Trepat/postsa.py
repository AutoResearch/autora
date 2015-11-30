import sys
import pandas as pd

sys.path.append('../../')
from mcmc import *

sys.path.append('../../Prior')
from fit_prior import read_prior_par

sys.path.append('../')
from data_analysis import post_SA

import iodata

VARS = [
    'CDH1',
    'CDH2',
    'CDH3',
    'ALPHACAT',
    'BETACAT',
    'P120',
    'ZO1',
]
Y = 'Sxx'
NS = 1000

if __name__ == '__main__':
    # Read the data
    inFileName = 'data/cadhesome_protein.csv'
    data, x, y = iodata.read_data(
        ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    print data

    prior_par = read_prior_par(sys.argv[1])
    print prior_par

    # SA
    outFileName = '__'.join((inFileName.split('/')[-1],
                             sys.argv[1].split('/')[-1]))
    t = post_SA(x, y, VARS, prior_par, npar=6, ns=2000, fn_label=outFileName,
                T_ini=1., T_fin=0.01, T_sched=0.95)
