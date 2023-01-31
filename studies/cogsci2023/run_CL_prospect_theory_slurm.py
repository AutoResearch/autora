import pickle
import time
import argparse
import numpy as np
from random import seed

from models.models import model_inventory
from utils import (
    fit_theorist,
    get_experimentalist,
    get_MSE,
    get_seed_experimentalist,
)

# ID = 3

# META PARAMETERS
num_cycles = 50  # number of cycles (20)
samples_for_seed = 10  # number of seed data_closed_loop points (20)
samples_per_cycle = 10  # number of data_closed_loop points chosen per cycle (20)
theorist_epochs = 500  # number of epochs for theorist (not used for logistic regression)
num_repetitions = 20

# SELECT THEORIST
theorist_name = "Logit Regression"

# SELECT GROUND TRUTH MODEL
ground_truth_name = "prospect_theory"  # OPTIONS: see models.py

experimentalists = [
    'popper',
    'falsification',
    'random',
    "dissimilarity",
    'model disagreement',
    'least confident',
]

st = time.time()

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()

repetitions = np.arange(num_repetitions)
experimentalist_ids = np.arange(len(experimentalists))
conditions = np.array(np.meshgrid(repetitions, experimentalist_ids)).T.reshape(-1,2)

rep = conditions[args.slurm_id,0] # args.slurm_id
experimentalist_id = conditions[args.slurm_id,1]
experimentalist_name = experimentalists[experimentalist_id]
seed(rep)

# SET UP STUDY
MSE_log = list()
cycle_log = list()
repetition_log = list()
theory_log = list()
conditions_log = list()
observations_log = list()
experimentalist_log = list()

# get information from the ground truth model
if ground_truth_name not in model_inventory.keys():
    raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
(metadata, data_fnc, experiment) = model_inventory[ground_truth_name]

# split data_closed_loop into training and test sets
X_full, y_full = data_fnc(metadata)
X_train = X_full.copy()
y_train = y_full.copy()
X_test = X_full.copy()
y_test = y_full.copy()

# get seed experimentalist
experimentalist_seed = get_seed_experimentalist(
    X_train, metadata, samples_for_seed
)

# generate seed data_closed_loop
X_seed = experimentalist_seed.run()
y_seed = experiment(X_seed)

# set up and fit theorist
print("Fitting theorist...")
theorist_seed = fit_theorist(X_seed, y_seed, theorist_name, metadata, theorist_epochs)

# log initial performance
MSE_log.append(get_MSE(theorist_seed, X_test, y_test))
cycle_log.append(0)
repetition_log.append(rep)
theory_log.append(theorist_seed)
conditions_log.append(X_seed)
observations_log.append(y_seed)
experimentalist_log.append(experimentalist_name)

X = X_seed.copy()
y = y_seed.copy()
theorist = theorist_seed

# now that we have the seed data_closed_loop and model, we can start the recovery loop
for cycle in range(num_cycles):
    print(f"Starting cycle {cycle} of {num_cycles}...")

    # generate experimentalist
    experimentalist = get_experimentalist(
        experimentalist_name,
        X,
        y,
        X_train,
        metadata,
        theorist,
        samples_per_cycle,
    )

    # get new experiment conditions
    st = time.time()
    print("Running experimentalist..." + experimentalist_name)
    X_new = experimentalist.run()

    et = time.time()
    elapsed_time = et - st
    print("Elapsed time: " + str(elapsed_time))

    # run experiment
    print("Running experiment...")
    y_new = experiment(X_new)

    # combine old and new data_closed_loop
    X = np.row_stack([X, X_new])
    y = np.row_stack([y, y_new])

    # fit theory
    print("Fitting theorist...")
    theorist = fit_theorist(X, y, theorist_name, metadata, theorist_epochs)

    # evaluate theory fit
    print("Evaluating fit...")
    MSE_log.append(get_MSE(theorist, X_test, y_test))
    cycle_log.append(cycle + 1)
    repetition_log.append(rep)
    theory_log.append(theorist)
    conditions_log.append(X)
    observations_log.append(y)
    experimentalist_log.append(experimentalist_name)

# save and load pickle file
file_name = (
    "data_closed_loop/"
    + ground_truth_name
    + "_"
    + theorist_name
    + "_"
    + experimentalist_name
    + "_"
    + str(rep)
    + ".pickle"
)

with open(file_name, "wb") as f:
    # simulation configuration
    configuration = dict()
    configuration["ground_truth"] = ground_truth_name
    configuration["theorist_name"] = theorist_name
    configuration["experimentalist_name"] = experimentalist_name
    configuration["num_cycles"] = num_cycles
    configuration["samples_per_cycle"] = samples_per_cycle
    configuration["samples_for_seed"] = samples_for_seed
    configuration["theorist_epochs"] = theorist_epochs
    configuration["repetitions"] = rep
    # configuration["test_size"] = test_size

    object_list = [
        configuration,
        MSE_log,
        cycle_log,
        repetition_log,
        theory_log,
        conditions_log,
        observations_log,
        experimentalist_log
    ]

    pickle.dump(object_list, f)

et = time.time()
elapsed_time = et - st
print(f"Elapsed time: {elapsed_time}")
