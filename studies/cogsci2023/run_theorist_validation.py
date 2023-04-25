import pickle
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from studies.cogsci2023.models.models import model_inventory
from studies.cogsci2023.utils import fit_theorist, get_MSE, get_LL, get_BIC, get_DL

# META PARAMETERS
repetitions = 1  # specifies how many times to repeat the study (20)
test_size = 0.2  # proportion of test set size to training set size

theorists = [
    "BMS",
    "BMS Fixed Root",
    "Logit Regression",
    "DARTS 3 Nodes",
    "MLP",
    "BSR"
]

for rep in range(repetitions):
    df = pd.read_csv('baseline_datasets/data_fleming')

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=rep
    )

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
        theorist = fit_theorist(X_train, y_train, theorist_name, metadata)
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
        bic = get_BIC(theorist, theorist_name, mse, num_obs)
        dl = get_DL(theorist, theorist_name, mse, num_obs)

        MSE_log.append(mse)
        LL_log.append(ll)
        BIC_log.append(bic)
        DL_log.append(dl)
        if hasattr(theorist, "model_") and 'BMS' not in theorist_name:
            theory_log.append(theorist.model_)
        elif 'BSR' in theorist_name:
            print('BSR not compatible with pickle')
            pass
        else:
            theory_log.append(theorist)
        theorist_name_log.append(theorist_name)

    # save and load pickle file
    file_name = "data_theorist/" + ground_truth_name + "_" + str(rep) + ".pickle"

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
