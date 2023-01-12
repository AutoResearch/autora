import warnings

import pandas as pd

from autora.skl.bms import BMSRegressor
from autora.theorist.bms import get_priors

warnings.filterwarnings("ignore")


# load data_closed_loop
XLABS = [
    "S1",
    "S2",
]

raw_data = pd.read_csv("weber_data.csv")
x, y = raw_data[XLABS], raw_data["difference_detected"]

# initialize model
# hyper parameters

prior_par, _ = get_priors()

# temperatures
ts = [1.0] + [1.04**k for k in range(1, 20)]

# epoch num
epochs = 1500

estimator = BMSRegressor(prior_par, ts, epochs)
estimator = estimator.fit(x, y)

print(estimator.model_)

test_x = x.head()
estimator.predict(test_x)

estimator.present_results()
