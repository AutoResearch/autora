# main file for Bayesian Scientist

# import packages
import sys
import warnings

import pandas as pd

import utils
from parallel import Parallel
from Prior.fit_prior import read_prior_par

warnings.filterwarnings("ignore")

sys.path.append("./")
sys.path.append("./Prior/")


# load data
"""
XLABS = [
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
]"""
XLABS = [
    "S1",
    "S2",
]
nv = len(XLABS)
nc = 1
epochs = 1500
# raw_data = pd.read_csv('Validation/LogYe/data/seymour.csv')
# x, y = raw_data[XLABS], np.log(raw_data['rec'])

raw_data = pd.read_csv("experiment_0_data.csv")
# x, y = raw_data[XLABS], np.log(raw_data['difference_detected'])
x, y = raw_data[XLABS], raw_data["difference_detected"]

# initialize model
# hyper parameters
prior_par = read_prior_par(
    "./Prior/final_prior_param_sq.named_equations.nv2.np2.2016-09-09 18:49:43.038278.dat"
)
# temperatures
Ts = [1.0] + [1.04**k for k in range(1, 20)]
# model
pms = Parallel(
    Ts,
    variables=XLABS,
    parameters=["a%d" % i for i in range(nc)],
    x=x,
    y=y,
    prior_par=prior_par,
)

# run model ---
model, model_len, desc_len = utils.run(pms, epochs)

# present results
utils.present_results(model, model_len, desc_len)

# test predictions
utils.predict(model, x, y)

"""
Places where changes needed to be made in order to incorporate our own priors
If we plan to give simple priors, then we simply need to give a csv, with the
operations and their respective probabilities

I believe all necessary changes can be made in just mcmc.py

If we plan to give different operations than what comes pre-included, we will
need to make a change to the accepted operations at line 22 in mcmc.py

If we plan to make recursive priors, then we will probably need to insert code
at line 468,1105,1133,1136,1175,1176 in mcmc.py
"""
