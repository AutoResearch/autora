# load data_closed_loop
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from studies.cogsci2023.models.models import model_inventory, plot_inventory


# set the path to the data_closed_loop directory
path = "data_theorist/"
# ground_truth_name = "weber_fechner"  # OPTIONS: see models.py
from_dataset = True

df_validation = pd.DataFrame()
full_theory_log = []
entry = 0

if not from_dataset:
    for gt_model in os.listdir('models/'):
        if gt_model == '__init__.py' or gt_model == 'models.py' or not gt_model.endswith('.py'):
            continue
        print(gt_model)
        ground_truth_name = str(gt_model)[0:-3]  # remove '.py' from file name
        print(ground_truth_name)
        (metadata, data_fnc, experiment) = model_inventory[ground_truth_name]
        X_full, y_full = data_fnc(metadata)
        full_n = len(X_full)

        # create an empty list to store the loaded pickle files
        loaded_pickles = []

        # iterate through all files in the data_closed_loop directory
        for file in os.listdir(path):
            # check if the file is a pickle file
            if file.endswith(".pickle") and file.startswith(ground_truth_name):
                # open the pickle file and load the contents
                with open(os.path.join(path, file), "rb") as f:
                    pickle_data = pkl.load(f)
                # append the loaded data_closed_loop to the list of loaded pickles
                loaded_pickles.append(pickle_data)

        # df_validation = pd.DataFrame()
        # full_theory_log = []
        # entry = 0

        # import the loaded pickles into data_closed_loop frame
        for pickle in loaded_pickles:

            configuration = pickle[0]
            MSE_log = pickle[1]
            theory_log = pickle[2]
            theorist_name_log = pickle[3]
            DL_log = pickle[5]
            BIC_log = pickle[6]
            LL_log = pickle[7]
            if len(theory_log) > 6:
                continue

            for idx in range(len(MSE_log)):

                print(len(theory_log))
                row = dict()
                row["Entry"] = entry
                row["Theorist"] = theorist_name_log[idx]
                row["Ground Truth"] = configuration["ground_truth_name"]
                row["Mean Squared Error"] = MSE_log[idx]
                test_size = configuration["test_size"]
                num_obs = test_size * full_n
                row['Description Length'] = DL_log[idx]
                row['Bayesian Information Criterion'] = BIC_log[idx]
                row['Log Likelihood'] = LL_log[idx]
                full_theory_log.append(theory_log[idx])
                df_validation = df_validation.append(row, ignore_index=True)
                entry = entry + 1

        # remove MSE outliers
        theorists = df_validation["Theorist"].unique()
        gts = df_validation["Ground Truth"].unique()
        for theorist in theorists:
            for gt in gts:
                # compute the mean MSE for each experimentalist in the data_closed_loop frame
                mean_MSE = df_validation[
                    (df_validation["Theorist"] == theorist)
                    & (df_validation["Ground Truth"] == gt)
                ]["Mean Squared Error"].mean()
                std_MSE = df_validation[
                    (df_validation["Theorist"] == theorist)
                    & (df_validation["Ground Truth"] == gt)
                ]["Mean Squared Error"].std()
                # remove all rows with MSE above the mean + 3 standard deviations
                df_validation = df_validation.drop(
                    df_validation[
                        (df_validation["Theorist"] == theorist)
                        & (df_validation["Ground Truth"] == gt)
                        & (df_validation["Mean Squared Error"] > mean_MSE + 3 * std_MSE)
                    ].index
                )

    # save and load pickle file
        file_name = "data_theorist/df_validation.pickle"
        with open(file_name, "wb") as f:
            pkl.dump(df_validation, f)
else:
    with open("data_theorist/df_validation.pickle", "rb") as f:
        df_validation = pkl.load(f)

# MSE Plot
g = sns.FacetGrid(df_validation, col="Ground Truth", col_wrap=4, ylim=(0, 1))
g.map(sns.barplot, "Theorist", "Mean Squared Error")
plt.show()

# Log Likelihood Plot
g = sns.FacetGrid(df_validation, col="Ground Truth", col_wrap=4, ylim=(0, -10))
g.map(sns.barplot, "Theorist", "Log Likelihood")
plt.show()

# Description Length Plot
g = sns.FacetGrid(df_validation, col="Ground Truth", col_wrap=4, ylim=(0, -10))
g.map(sns.barplot, "Theorist", "Description Length")
plt.show()

if __name__ == '__main__':
    ...