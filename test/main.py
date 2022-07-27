# main file for Bayesian Scientist

# import packages
import warnings

import pandas as pd
import utils

# from fit_prior import read_prior_par
from parallel import Parallel

warnings.filterwarnings("ignore")


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
epochs = 300
# raw_data = pd.read_csv('Validation/LogYe/data/seymour.csv')
# x, y = raw_data[XLABS], np.log(raw_data['rec'])

raw_data = pd.read_csv("aer_bms/experiment_0_data.csv")
# x, y = raw_data[XLABS], np.log(raw_data['difference_detected'])
x, y = raw_data[XLABS], raw_data["difference_detected"]

# initialize model
# hyper parameters
# prior_par = read_prior_par(
#    "./Prior/final_prior_param_sq.named_equations.nv2.np2.2016-09-09 18:49:43.038278.dat"
# )

prior_par = {
    "Nopi_/": 5.912205942815285,
    "Nopi_cosh": 8.12720511103694,
    "Nopi_-": 3.350846072163632,
    "Nopi_sin": 5.965917796154835,
    "Nopi_tan": 8.127427922862411,
    "Nopi_tanh": 7.799259068142255,
    "Nopi_**": 6.4734429542245495,
    "Nopi_pow2": 3.3017352779079734,
    "Nopi_pow3": 5.9907496760026175,
    "Nopi_exp": 4.768665265735502,
    "Nopi_log": 4.745957377206544,
    "Nopi_sqrt": 4.760686909134266,
    "Nopi_cos": 5.452564657261127,
    "Nopi_sinh": 7.955723540761046,
    "Nopi_abs": 6.333544134938385,
    "Nopi_+": 5.808163661224514,
    "Nopi_*": 5.002213595420244,
    "Nopi_fac": 10.0,
    "Nopi2_*": 1.0,
}
# Prior.fit_prior()


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
