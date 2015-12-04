import sys
import pandas as pd

sys.path.append('../../')
from mcmc import *

sys.path.append('../../Prior')
from fit_prior import read_prior_par

sys.path.append('../')
from data_analysis import model_averaging_valid

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
NS = 100

if __name__ == '__main__':
    # Read the data
    inFileName = 'data/cadhesome_protein.csv'
    data, x, y = iodata.read_data(
        ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    print data

    priorFileName = sys.argv[1]
    prior_par = read_prior_par(priorFileName)
    print prior_par

    # Model averaging
    npar = priorFileName[priorFileName.find('.np') + 3:]
    npar = int(npar[:npar.find('.')])
    print 'NPAR =', npar
    mse, mae = model_averaging_valid(x, y, VARS, prior_par, npar=npar, ns=NS,
                                     method='lko', k=1)
    print 'RMSE:', np.sqrt(np.mean(mse)), mse
    print 'MAE: ', np.mean(mae), mae
