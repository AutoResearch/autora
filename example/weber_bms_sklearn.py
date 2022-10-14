import warnings

import pandas as pd

from autora_bms.prior import get_priors
from autora_bms.skl.bms import BMSRegressor

warnings.filterwarnings("ignore")


# load data
XLABS = [
    "S1",
    "S2",
]

raw_data = pd.read_csv("experiment_0_data.csv")
x, y = raw_data[XLABS], raw_data["difference_detected"]

# initialize model
# hyper parameters

prior_par, _ = get_priors()

# temperatures
ts = [1.0] + [1.04**k for k in range(1, 20)]

# epoch num
epochs = 100

estimator = BMSRegressor(prior_par, ts, epochs)
estimator = estimator.fit(x, y)

print(estimator.model_)

test_x = x.head()
estimator.predict(test_x)


"""
Places where changes needed to be made in order to incorporate our own priors
If we plan to give simple priors, then we simply need to give a csv, with the
operations and their respective probabilities

I believe all necessary changes can be made in mcmc.py

If we plan to give different operations than what comes pre-included, we will
need to make a change to the accepted operations at line 22 in mcmc.py

If we plan to make recursive priors, then we will probably need to insert code
at line 468,1105,1133,1136,1175,1176 in mcmc.py
"""
