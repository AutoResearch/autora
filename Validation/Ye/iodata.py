import sys
import pandas as pd

XVARS = [
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


def read_data(ylabel='rec', xlabels=XVARS, in_fname='data/seymour.csv'):
    data = pd.read_csv(in_fname)
    
    ### Divide discharges by 1,000 to make the scale more manageable
    ##for v in XVARS:
    ##    if v.startswith('D_'):
    ##        data[v] /= 1000.

    x = data[xlabels]
    y = data[ylabel]
    return data, x, y

if __name__ == '__main__':
    d, x, y = read_data()
    
    print x
    print y
