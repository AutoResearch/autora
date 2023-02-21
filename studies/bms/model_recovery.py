import pickle

from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from studies.bms.models_cache.recoverable.exp_learning import exp_learning_experiment
from studies.bms.models_cache.recoverable.weber import weber_experiment

# recoverable models
# make & collect data
data_recoverable = dict()
data_recoverable["weber"] = weber_experiment()
data_recoverable["exp_learning"] = exp_learning_experiment()

# initialize theorists
bms = BMSRegressor(epochs=5)
darts = DARTSRegressor(max_epochs=5)
theorists = [bms, darts]

# attempt to recover recoverable models
recoverable = dict()

for key in data_recoverable.keys():
    for theorist in theorists:
        X, y = data_recoverable[key]
        theorist = theorist.fit(X, y).model_
        recoverable[key + "_" + theorist.__name__] = {
            "data": data_recoverable[key],
            "theorist_name": theorist.__name__,
            "theory": theorist.model_,
        }

# non-recoverable models
# make & collect data
data_nonrecoverable = dict()
data_nonrecoverable["prospect_theory"] = weber_experiment()

# initialize theorists
bms = BMSRegressor(epochs=5)
darts = DARTSRegressor(max_epochs=5)
theorists = [bms, darts]

# attempt to recover/approximate non-recoverable models
nonrecoverable = dict()

for key in data_nonrecoverable.keys():
    for theorist in theorists:
        X, y = data_nonrecoverable[key]
        theorist = theorist.fit(X, y)
        nonrecoverable[key + "_" + theorist.__name__] = {
            "data": data_nonrecoverable[key],
            "theorist_name": theorist.__name__,
            "theory": theorist.model_,
        }

# save and load pickle file
file_name = "data/data.pickle"

with open(file_name, "wb") as f:
    # simulation recovery
    object_list = [recoverable, nonrecoverable]

    pickle.dump(object_list, f)

if __name__ == "__main__":
    ...
