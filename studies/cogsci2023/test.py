import pickle
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed

from models.models import model_inventory
from utils import (
    fit_theorist,
    get_experimentalist,
    get_MSE,
    get_seed_experimentalist,
)

# META PARAMETERS
num_cycles = 20  # number of cycles (20)
samples_for_seed = 100  # number of seed data_closed_loop points (20)
samples_per_cycle = 100  # number of data_closed_loop points chosen per cycle (20)
theorist_epochs = 1500  # number of epochs for BMS (500)

# SELECT THEORIST
theorist_name = "BMS"

# SELECT GROUND TRUTH MODEL
ground_truth_name = "prospect_theory"  # OPTIONS: see models.py

experimentalist_name = 'random'

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
print("Fitting seed theorist...")
theorist_seed = fit_theorist(X_seed, y_seed, theorist_name, metadata, theorist_epochs)
seed_MSE = get_MSE(theorist_seed, X_test, y_test)

X = X_seed.copy()
y = y_seed.copy()
theorist = theorist_seed

for cycle in range(num_cycles):
    print(f"Cycle {cycle + 1} of {num_cycles}...")
    # get experimentalist
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
    print("Running experimentalist...")
    X_new = experimentalist.run()

    # run experiment
    print("Running experiment...")
    y_new = experiment(X_new)

    # combine old and new data_closed_loop
    X = np.row_stack([X, X_new])
    y = np.row_stack([y, y_new])

final_theorist = fit_theorist(X, y, theorist_name, metadata, theorist_epochs)
final_MSE = get_MSE(final_theorist, X_test, y_test)

print("X seed shape")
print(X_seed.shape)
print("X final shape")
print(X.shape)

print(f"Seed theorist MSE: {seed_MSE}")
print(f"Final theorist MSE: {final_MSE}")


