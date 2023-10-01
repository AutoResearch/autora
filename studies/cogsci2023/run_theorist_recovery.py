import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split

from studies.cogsci2023.models.models import model_inventory, param_dict
from studies.cogsci2023.utils import fit_theorist, get_MSE, get_LL, get_BIC, get_DL

# META PARAMETERS
repetitions = 1  # specifies how many times to repeat the study (20)
seed = 20
test_size = 0.2  # proportion of test set size to training set size
noise = 0.5

# SELECT GROUND TRUTH MODEL
ground_truth_name = "shepard_luce_choice"  # OPTIONS: see models.py

# theorists = [
#     'BMS',
#     'BMS Prior Default',
#     'BMS Prior Williams2023Psychophysics',
#     'BMS Prior Williams2023CognitivePsychology',
#     'BMS Prior Williams2023BehavioralEconomics',
# ]

theorists = [
    "BMS"
    # 'Regression',
    # 'MLP'
    ]

for rep in range(repetitions):

    # get information from the ground truth model
    if ground_truth_name not in model_inventory.keys():
        raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
    (metadata, data_fnc, experiment) = model_inventory[ground_truth_name]

    # split data_closed_loop into training and test sets
    X_full, y_full = data_fnc(metadata, std=noise)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=rep
    )

    num_var = X_train.shape[1]
    num_param = param_dict[ground_truth_name]

    MSE_log = list()
    LL_log = list()
    BIC_log = list()
    DL_log = list()
    theory_log = list()
    theorist_name_log = list()
    elapsed_time_log = list()
    print(X_train)
    for theorist_name in theorists:

        # fit the theorist
        st = time.time()
        theorist = fit_theorist(X_train, y_train, theorist_name, metadata, None, num_param, num_var)
        et = time.time()
        elapsed_time = et - st
        elapsed_time_log.append(elapsed_time)

        # if theorist_name == "BMS" or theorist_name == "BMS Fixed Root" or theorist_name == 'Logit Regression'\
        #    or 'DARTS' in theorist_name:
        #    DL = get_DL(theorist, theorist_name, get_MSE(theorist, X_test, y_test), len(y_test))
        # else:
        #    DL = 0
        # DL_log.append(DL)
        mse = get_MSE(theorist, X_test, y_test)
        ll = get_LL(mse)
        num_obs = X_test.shape[0]
        bic = get_BIC(theorist, theorist_name, mse, num_obs, num_param, num_var)
        dl = get_DL(theorist, theorist_name, mse, num_obs, num_param, num_var)

        MSE_log.append(mse)
        LL_log.append(ll)
        BIC_log.append(bic)
        DL_log.append(dl)
        # if hasattr(theorist, "model_") and 'BMS' not in theorist_name:
        #     theory_log.append(theorist.model_)
        # elif 'BSR' in theorist_name:
        #     print('BSR not compatible with pickle')
        #     pass
        # else:
        #     theory_log.append(theorist)
        # theorist_name_log.append(theorist_name)

        if 'BMS' in theorist_name:
            eq = str(theorist.model_)
            for par in theorist.model_.par_values['d0'].keys():
                eq = eq.replace(par, str(np.round(np.float64(theorist.model_.par_values['d0'][par]),
                                                  decimals=2)))
            theory_log.append(eq)

        theorist_name_log.append(theorist_name)

    # save and load pickle file
    file_name = "data_noise_"+str(noise)+"/" + ground_truth_name + "_" + str(rep+seed) + ".pickle"


    with open(file_name, "wb") as f:
        # simulation configuration
        configuration = dict()
        configuration["ground_truth_name"] = ground_truth_name
        configuration["repetitions"] = repetitions
        configuration["test_size"] = test_size

        object_list = [
            configuration,
            MSE_log,
            theory_log,
            theorist_name_log,
            elapsed_time_log,
            DL_log,
            BIC_log,
            LL_log,
        ]

        pickle.dump(object_list, f)

# using datetime module
import datetime

# ct stores current time
ct = datetime.datetime.now()
print("current time:-", ct)
