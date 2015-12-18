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
    'v1',
    'v2',
    'v3',
    'v4',
    #'v5',
    'v6',
    #'v7',
    'v8',
    #'v9',
    'v10',
    'v11',
    'v12',
    'v13',
    'v14',
    'v15',
    'v16',
]
Y = 'Barrier'
NS = 100

if __name__ == '__main__':
    # Read the data
    inFileName = 'data/H_features.dat'
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
    mse, mae = model_averaging_valid(x, y, VARS, prior_par, npar=npar,
                                     ns=NS, thin=200,
                                     method='lko', k=1)
    print 'RMSE:', np.sqrt(np.mean(mse)), mse
    print 'MAE: ', np.mean(mae), mae
