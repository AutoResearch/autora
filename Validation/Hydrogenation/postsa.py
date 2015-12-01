import sys
import pandas as pd

sys.path.append('../../')
from mcmc import *

sys.path.append('../../10-Prior')
from fit_prior import read_prior_par

sys.path.append('../')
from data_analysis import post_SA

import iodata

VARS = [
    'v1',
    'v2',
    'v3',
    'v4',
    'v5',
    'v6',
    'v7',
#    'v8',
    'v9',
#    'v10',
    'v11',
    'v12',
#    'v13',
    'v14',
    'v15',
#    'v16',
]
Y = 'Barrier'
NS = 1000

if __name__ == '__main__':
    # Read the data
    inFileName = 'H_features.dat'
    data, x, y = iodata.read_data(
        ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    print data

    prior_par = read_prior_par(sys.argv[1])
    print prior_par

    # SA
    t = post_SA(x, y, VARS, prior_par, npar=6, ns=2000, fn_label=inFileName,
                T_ini=1., T_fin=0.01, T_sched=0.95)
