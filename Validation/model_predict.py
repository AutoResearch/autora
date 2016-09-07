import sys
import ast
import numpy as np
from scipy.stats import sem
from sklearn import cross_validation
from sympy import sympify, latex
from optparse import OptionParser

sys.path.append('..')
from mcmc import Tree
from iodata import *

sys.path.append('../Prior')
from fit_prior import read_prior_par


# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] DATASET MODELSTR')
    parser.add_option("-v", "--parvalues", dest="par_values", default=None,
                      help="Use these parameter values (in dictionary format)")
    return parser

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Command line parameters
    parser = parse_options()
    opt, args = parser.parse_args()
    if opt.par_values != None:
        opt.par_values = ast.literal_eval(opt.par_values)
    dset = args[0]
    modelstr = args[1]

    # Load the data
    data, x, y = read_data(dset)

    # Build the tree and set parameter values, if necessary
    t = Tree(from_string=modelstr)
    t.x, t.y = x, y
    if opt.par_values != None:
        t.par_values = opt.par_values
        t.get_sse(fit=False)
    else:
        t.get_sse(fit=True)

    # Make the prediction
    ypred = t.predict(x)
    x['yreal'] = y
    x['ypred'] = ypred
    x.to_csv('%s/model_predict.csv' % dset)
