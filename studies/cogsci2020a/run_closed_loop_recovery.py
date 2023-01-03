from models import model_inventory
from sklearn.model_selection import train_test_split
from studies.cogsci2020a.utils import get_theorist, get_seed_experimentalist, get_experimentalist, \
    get_MSE
import numpy as np
import pickle

# META PARAMETERS
num_cycles = 5                  # number of cycles
samples_for_seed = 10           # number of seed data points
samples_per_cycle = 10           # number of data points chosen per cycle
theorist_epochs = 500            # number of epochs for BMS
repetitions = 10                 # specifies how many times to repeat the study
test_size = 0.20                # proportion of test set size to training set size

# SELECT THEORIST
# OPTIONS: BMS, DARTS
theorist_name  = "BMS"

# SELECT GROUND TRUTH MODEL
# OPTIONS: weber_fechner, exp_learning, stroop_model
ground_truth_name = "weber_fechner"

# SELECT EXPERIMENTALIST
# OPTIONS: random, dissimilarity, popper, model disagreement
experimentalists = ['random',
                    # 'dissimilarity',
                    # 'popper',
                    # 'model disagreement',
                    ]



for experimentalist_name in experimentalists:

    # SET UP STUDY
    MSE_log = list()
    cycle_log = list()
    repetition_log = list()
    theory_log = list()
    for rep in range(repetitions):

        # get information from the ground truth model
        if ground_truth_name not in model_inventory.keys():
            raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
        (metadata, data, experiment) = model_inventory[ground_truth_name]

        # split data into training and test sets
        X_full, y_full = data
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                            test_size=test_size,
                                                            random_state=rep)

        # get seed experimentalist
        experimentalist_seed = get_seed_experimentalist(X_train,
                                                        metadata,
                                                        samples_for_seed)

        # generate seed data
        X = experimentalist_seed.run()
        y = experiment(X)

        # set up theorist
        theorist = get_theorist(theorist_name, theorist_epochs)

        # derive seed model
        found_theory = False
        while not found_theory:
            try:
                theorist.fit(X, y)
                found_theory = True
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                print("Trying again....")

        # log initial performance
        MSE_log.append(get_MSE(theorist.model_, X_test, y_test))
        cycle_log.append(0)
        repetition_log.append(rep)
        theory_log.append(theorist.model_)

        # now that we have the seed data and model, we can start the recovery loop
        for cycle in range(num_cycles):

            # generate experimentalist
            experimentalist = get_experimentalist(experimentalist_name,
                                                  X,
                                                  y,
                                                  X_train,
                                                  metadata,
                                                  theorist,
                                                  samples_per_cycle)

            # get new experiment conditions
            X_new = experimentalist.run()

            # run experiment
            y_new = experiment(X_new)

            # combine old and new data
            X = np.row_stack([X, X_new])
            y = np.row_stack([y, y_new])

            # fit theory
            found_theory = False

            while not found_theory:
                try:
                    theorist.fit(X, y)
                    found_theory = True
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    print("Trying again....")

            # evaluate theory fit
            MSE_log.append(get_MSE(theorist.model_, X_test, y_test))
            cycle_log.append(cycle+1)
            repetition_log.append(rep)
            theory_log.append(theorist.model_)

    # save and load pickle file
    file_name = "data/" + ground_truth_name + "_" + theorist_name +  "_" + experimentalist_name + ".pickle"

    with open(file_name, 'wb') as f:
        # simulation configuration
        configuration = dict()
        configuration["theorist_name"] = theorist_name
        configuration["experimentalist_name"] = experimentalist_name
        configuration["num_cycles"] = num_cycles
        configuration["samples_per_cycle"] = samples_per_cycle
        configuration["theorist_epochs"] = theorist_epochs
        configuration["repetitions"] = repetitions
        configuration["test_size"] = test_size

        object_list = [configuration,
                       MSE_log,
                       cycle_log,
                       repetition_log,
                       theory_log]

        pickle.dump(object_list, f)









