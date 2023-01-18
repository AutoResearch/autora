from autora.model.inventory import retrieve
from studies.cogsci2023.models.models import model_inventory
from sklearn.model_selection import train_test_split
from studies.cogsci2023.utils import fit_theorist, get_MSE
import pickle
import time

# META PARAMETERS
repetitions = 20                 # specifies how many times to repeat the study (20)
test_size = 0.2                # proportion of test set size to training set size

# SELECT GROUND TRUTH MODEL
ground_truth_name = "expected_value" # OPTIONS: see models.py

theorists = [
             'MLP',
             'DARTS 2 Nodes',
             'DARTS 3 Nodes',
             'BMS',
             'BMS Fixed Root'
             ]

for rep in range(repetitions):

    # get information from the ground truth model
    if ground_truth_name not in model_inventory.keys():
        raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
    (metadata, data_fnc, experiment) = retrieve(ground_truth_name, kind="model:v0")

    # split data_closed_loop into training and test sets
    X_full, y_full = data_fnc(metadata)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                        test_size=test_size,
                                                        random_state=rep)

    MSE_log = list()
    theory_log = list()
    theorist_name_log = list()
    elapsed_time_log = list()

    for theorist_name in theorists:

        # fit the theorist
        st = time.time()
        theorist = fit_theorist(X_train, y_train, theorist_name, metadata)
        et = time.time()
        elapsed_time = et - st
        elapsed_time_log.append(elapsed_time)

        MSE_log.append(get_MSE(theorist, X_test, y_test))
        if hasattr(theorist, 'model_'):
            theory_log.append(theorist.model_)
        else:
            theory_log.append(theorist)
        theorist_name_log.append(theorist_name)

    # save and load pickle file
    file_name = "data_theorist/" + ground_truth_name + "_" + str(rep) + ".pickle"

    with open(file_name, 'wb') as f:
        # simulation configuration
        configuration = dict()
        configuration["ground_truth_name"] = ground_truth_name
        configuration["repetitions"] = repetitions
        configuration["test_size"] = test_size

        object_list = [configuration,
                       MSE_log,
                       theory_log,
                       theorist_name_log,
                       elapsed_time_log,
                       ]

        pickle.dump(object_list, f)

# using datetime module
import datetime
# ct stores current time
ct = datetime.datetime.now()
print("current time:-", ct)
