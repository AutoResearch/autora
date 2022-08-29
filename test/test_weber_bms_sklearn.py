import warnings

import pandas as pd

import sys
sys.path.append("./")

from aer_bms.skl.bms import BMS

warnings.filterwarnings("ignore")


# load data
XLABS = [
    "S1",
    "S2",
]

epochs = 300

raw_data = pd.read_csv("test/experiment_0_data.csv")
x, y = raw_data[XLABS], raw_data["difference_detected"]

# initialize model
# hyper parameters

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

# temperatures
ts = [1.0] + [1.04**k for k in range(1, 20)]

# epoch num
epochs = 100

estimator = BMS(prior_par, ts, epochs)
estimator = estimator.fit(x, y)

print(estimator.model_)

test_x = x.head()
estimator.predict(test_x)