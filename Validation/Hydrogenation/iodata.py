import sys
import pandas as pd

XVARS = [
    'v1',
    'v2',
    'v3',
    'v4',
    'v5',
    'v6',
    'v7',
    'v8',
    'v9',
    'v10',
    'v11',
    'v12',
    'v13',
    'v14',
    'v15',
    'v16',
]


def read_data(ylabel='Barrier', xlabels=XVARS, in_fname='data/H_features.dat'):
    data = pd.read_csv(in_fname)
    x = data[xlabels]
    y = data[ylabel]
    return data, x, y

if __name__ == '__main__':
    d, x, y = read_data()
    
    print x
    print y
