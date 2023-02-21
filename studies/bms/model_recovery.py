import pickle

from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from studies.bms.models_cache.recoverable.exp_learning import exp_learning_experiment
from studies.bms.models_cache.recoverable.weber import weber_experiment
from studies.bms.utils import mountain, threshold2, threshold3

# recoverable models
# make & collect data
data_recoverable = dict()
data_recoverable["weber"] = weber_experiment()
# data_recoverable["exp_learning"] = exp_learning_experiment()
# data_recoverable["expected_value"]
# data_recoverable["stevens_power_law"]
# data_recoverable["tva"]

# initialize theorists
bms = BMSRegressor(epochs=5)
darts = DARTSRegressor(max_epochs=5)
theorists = [bms, darts]

# attempt to recover recoverable models
recoverable = dict()

for key in data_recoverable.keys():
    for theorist in theorists:
        X, y = data_recoverable[key]
        if theorist.__name__ == "BMSRegressor":
            theorist = theorist.fit(X, y, custom_ops=[mountain, threshold2, threshold3])
        else:
            theorist = theorist.fit(X, y)
        recoverable[key + "_" + theorist.__name__] = {
            "data": data_recoverable[key],
            "theorist_name": theorist.__name__,
            "theory": theorist.model_,
        }

# non-recoverable models
# make & collect data
data_nonrecoverable = dict()
# data_nonrecoverable["evc_coged"]
# data_nonrecoverable["evc_congruency"]
# data_nonrecoverable["evc_demand_selection"]
# data_nonrecoverable["prospect_theory"]
# data_nonrecoverable["shepard_luce_choice"]
# data_nonrecoverable["stroop_model"]
# data_recoverable["task_switching"]


# initialize theorists
bms = BMSRegressor(epochs=5)
darts = DARTSRegressor(max_epochs=5)
theorists = [bms, darts]

# attempt to recover/approximate non-recoverable models
nonrecoverable = dict()

for key in data_nonrecoverable.keys():
    for theorist in theorists:
        X, y = data_nonrecoverable[key]
        if theorist.__name__ == "BMSRegressor":
            theorist = theorist.fit(X, y, custom_ops=[mountain, threshold2, threshold3])
        else:
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
