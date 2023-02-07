import pickle

import pandas as pd

from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from studies.bms.models.context_free.weber import weber_experiment
from studies.bms.models.context_free.exp_learning import exp_learning_experiment


# make & collect data
data = dict()
data['weber'] = weber_experiment()
data['exp_learning'] = exp_learning_experiment()

# record theorist model recovery attempts
recovery = dict()
theorist = dict()
bms = BMSRegressor(epochs=5)
darts = DARTSRegressor(max_epochs=5)
for key in data.keys():
    X, y = data[key]
    theorist['BMS'] = bms.fit(X, y).model_
    theorist['DARTS'] = darts.fit(X, y).model_
    recovery[key] = {
        'data': data[key],
        'theorist': theorist
    }

# save and load pickle file
file_name = "data/data.pickle"

with open(file_name, "wb") as f:
    # simulation recovery
    object_list = [recovery]

    pickle.dump(object_list, f)

if __name__ == "__main__":
    ...
