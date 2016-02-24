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
     'Hydrogenation' : [
            'Barrier',
            'DE',
            'NC',
            'diH',
            'Tr(R)',
            'Tr(P)',
            'DTr',
            'det(R)',
            'det(P)',
            'det(TS)',
            'RxP',
            'RxTS',
            'rank(R)',
            'rank(P)',
            'rank(TS)',
            'SpecRad(R)',
            'SpecRad(P)',
            'Type',
            'ID'
    ]
}
YLABS = {
    'Trepat' : 'Sxx',
    'Ye'     : 'rec',
    'Hydrogenation':'Barrier',
}
FNAMES = {
    'Trepat' : 'cadhesome_protein.csv',
    'Ye'     : 'seymour.csv',
    'Hydrogenation':'H_features.csv',
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
    if dset == 'Trepat' or dset == 'Ye' or dset=='Hydrogenation':
        data = pd.read_csv(in_fname)
        x = data[xlabels]
        y = data[ylabel]
    else:
        raise

    # Done
    return data, x, y
