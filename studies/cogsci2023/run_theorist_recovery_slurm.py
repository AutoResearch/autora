from models.models import model_inventory
from sklearn.model_selection import train_test_split

from utils import (
    fit_theorist,
    get_DL,
    get_MSE,
    get_BIC,
    get_LL
)

from random import seed
import numpy as np
import argparse
import pickle
import time

# META PARAMETERS
num_repetitions = 20                 # specifies how many times to repeat the study (20)
test_size = 0.2                # proportion of test set size to training set size

theorists = [
             'MLP',
             'DARTS 2 Nodes',
             'DARTS 3 Nodes',
             'Regression',
             'BMS',
             'BMS Fixed Root',
             'BMS Code Ops',
             'BMS No Prior',
             'BMS No Penalty',
             'BMS Lite',
             'BSR'
             ]

gts = ['weber_fechner',
       'stevens_power_law',
       'exp_learning',
       'expected_value',
       'prospect_theory',
       'stroop_model',
       'evc_demand_selection'
       ]

parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()

repetitions = np.arange(num_repetitions)
gt_ids = np.arange(len(gts))
conditions = np.array(np.meshgrid(repetitions, gt_ids)).T.reshape(-1,2)

rep = conditions[args.slurm_id,0] # args.slurm_id
gt_id = conditions[args.slurm_id,1]
ground_truth_name = gts[gt_id]
seed(rep)

# get information from the ground truth model
if ground_truth_name not in model_inventory.keys():
    raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
(metadata, data_fnc, experiment) = model_inventory[ground_truth_name]

# split data_closed_loop into training and test sets
X_full, y_full = data_fnc(metadata)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                    test_size=test_size,
                                                    random_state=rep)
MSE_log = list()
DL_log = list()
BIC_log = list()
LL_log = list()
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

    mse = get_MSE(theorist, X_test, y_test)
    ll = get_LL(mse)
    num_obs = X_test.shape[0]
    bic = get_BIC(theorist, theorist_name, mse, num_obs)
    dl = get_DL(theorist, theorist_name, mse, num_obs)

    MSE_log.append(mse)
    LL_log.append(ll)
    BIC_log.append(bic)
    DL_log.append(dl)
    if hasattr(theorist, 'model_') and 'BMS' not in theorist_name:
        theory_log.append(theorist.model_)
    elif 'BSR' in theorist_name:
        print('BSR not compatible with pickle')
        pass
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
                   DL_log,
                   BIC_log,
                   LL_log
                   ]

    pickle.dump(object_list, f)

# using datetime module
import datetime
# ct stores current time
ct = datetime.datetime.now()
print("current time:-", ct)
