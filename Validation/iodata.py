import sys
import pandas as pd



XVARS = {
    'Trepat' : [
        'CDH1',
        'CDH2',
        'CDH3',
        'ALPHACAT',
        'BETACAT',
        'P120',
        'ZO1',
    ],
    'Ye'     : [
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
    ],
}
YLABS = {
    'Trepat' : 'Sxx',
    'Ye'     : 'rec',
}
FNAMES = {
    'Trepat' : 'cadhesome_protein.csv',
    'Ye'     : 'seymour.csv',
}

def read_data(dset, ylabel=None, xlabels=None, in_fname=None):
    # Default values
    if ylabel == None:
        ylabel = YLABS[dset]
    if xlabels == None:
        xlabels = XVARS[dset]
    if in_fname == None:
        in_fname = '%s/data/%s' % (dset, FNAMES[dset])
    # Read 
    if dset == 'Trepat' or dset == 'Ye':
        data = pd.read_csv(in_fname)
        x = data[xlabels]
        y = data[ylabel]
    else:
        raise

    # Done
    return data, x, y
