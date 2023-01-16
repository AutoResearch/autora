import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from studies.cogsci2023.models.models import model_inventory
from studies.cogsci2023.utils import (
    fit_theorist,
    get_experimentalist,
    get_MSE,
    get_seed_experimentalist,
)

# META PARAMETERS
num_cycles = 20  # number of cycles (20)
samples_for_seed = 20  # number of seed data_closed_loop points (20)
samples_per_cycle = 20  # number of data_closed_loop points chosen per cycle (20)
theorist_epochs = 500  # number of epochs for BMS (500)
repetitions = 3  # specifies how many times to repeat the study (20)

# what I learned
# - increasing model noise doesn't help, it just puts an upper limit on the final validation error

# todo: add back noise to prospect theory
# todo: make sure resolution of all probed models is comparable (and potentially higher)

# SELECT THEORIST
# OPTIONS: BMS, DARTS
theorist_name = "BMS Fixed Root"

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

for experimentalist_name in experimentalists:

    # SET UP STUDY
    MSE_log = list()
    cycle_log = list()
    repetition_log = list()
    theory_log = list()
    conditions_log = list()
    observations_log = list()
    for rep in range(repetitions):

        # get information from the ground truth model
        if ground_truth_name not in model_inventory.keys():
            raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
        (metadata, data_fnc, experiment) = model_inventory[ground_truth_name]

        # split data_closed_loop into training and test sets
        X_full, y_full = data_fnc(metadata)
        # X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
        #                                                     test_size=test_size,
        #                                                     random_state=rep)
        # Since we know the GT, we can use the full dataset for training

        # get seed experimentalist
        experimentalist_seed = get_seed_experimentalist(
            X_full, metadata, samples_for_seed
        )

        # generate seed data_closed_loop
        X = experimentalist_seed.run()
        y = experiment(X)

        # set up and fit theorist
        print("Fitting theorist...")
        theorist = fit_theorist(X, y, theorist_name, metadata, theorist_epochs)

        # log initial performance
        MSE_log.append(get_MSE(theorist, X_full, y_full))
        cycle_log.append(0)
        repetition_log.append(rep)
        theory_log.append(theorist)
        conditions_log.append(X)
        observations_log.append(y)

        # now that we have the seed data_closed_loop and model, we can start the recovery loop
        for cycle in range(num_cycles):

            # generate experimentalist
            experimentalist = get_experimentalist(
                experimentalist_name,
                X,
                y,
                X_full,
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

            # fit theory
            print("Fitting theorist...")
            theorist = fit_theorist(X, y, theorist_name, metadata, theorist_epochs)

            # evaluate theory fit
            print("Evaluating fit...")
            MSE_log.append(get_MSE(theorist, X_full, y_full))
            cycle_log.append(cycle + 1)
            repetition_log.append(rep)
            theory_log.append(theorist)
            conditions_log.append(X)
            observations_log.append(y)

    # save and load pickle file
    file_name = (
        "data_closed_loop/"
        + ground_truth_name
        + "_"
        + theorist_name
        + "_"
        + experimentalist_name
        + ".pickle"
    )

    with open(file_name, "wb") as f:
        # simulation configuration
        configuration = dict()
        configuration["theorist_name"] = theorist_name
        configuration["experimentalist_name"] = experimentalist_name
        configuration["num_cycles"] = num_cycles
        configuration["samples_per_cycle"] = samples_per_cycle
        configuration["samples_for_seed"] = samples_for_seed
        configuration["theorist_epochs"] = theorist_epochs
        configuration["repetitions"] = repetitions
        # configuration["test_size"] = test_size

        object_list = [
            configuration,
            MSE_log,
            cycle_log,
            repetition_log,
            theory_log,
            conditions_log,
            observations_log,
        ]

        pickle.dump(object_list, f)
