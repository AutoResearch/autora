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
    'eff',
    'D_max',
    'D_apr',
    'D_may',
    'D_jun',
    'ET_apr',
    'ET_may',
    'ET_jun',
    'PT_apr',
    'PT_may',
    'PT_jun',
    'PT_jul',
    'PDO_win',
]
Y = 'rec'
NS = 1000

if __name__ == '__main__':
    # Read the data
    inFileName = 'seymour.csv'
    data, x, y = iodata.read_data(
        ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    print data

    prior_par = read_prior_par(sys.argv[1])
    print prior_par

    # SA
    t = post_SA(x, y, VARS, prior_par, npar=6, ns=200, fn_label=inFileName,
                T_ini=2., T_fin=0.01, T_sched=0.95)
