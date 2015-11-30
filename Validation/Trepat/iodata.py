import sys
import pandas as pd

XVARS = [
    'CDH1',
    'CDH2',
    'CDH3',
    'BETACAT',
    'P120',
]

def read_data(ylabel='Sxx', xlabels=XVARS,
              in_fname='data/cadhesome_protein.csv'):
    data = pd.read_csv(in_fname)
    x = data[xlabels]
    y = data[ylabel]
    return data, x, y
