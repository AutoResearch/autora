# load data_closed_loop
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from autora.model.inventory import retrieve
from studies.cogsci2023.models.models import model_inventory, plot_inventory

# set the path to the data_closed_loop directory
path = "data_theorist/"
ground_truth_name = "expected_value"  # OPTIONS: see models.py
plot_theorist = "BMS Fixed Root"

# create an empty list to store the loaded pickle files
loaded_pickles = []

# iterate through all files in the data_closed_loop directory
for file in os.listdir(path):
    # check if the file is a pickle file
    if file.endswith(".pickle") and file.startswith(ground_truth_name):
        # open the pickle file and load the contents
        with open(os.path.join(path, file), "rb") as f:
            pickle_data = pickle.load(f)
        # append the loaded data_closed_loop to the list of loaded pickles
        loaded_pickles.append(pickle_data)

df_validation = pd.DataFrame()
full_theory_log = []
entry = 0

# import the loaded pickles into data_closed_loop frame
for pickle in loaded_pickles:

    configuration = pickle[0]
    MSE_log = pickle[1]
    theory_log = pickle[2]
    theorist_name_log = pickle[3]

    for idx in range(len(MSE_log)):

        row = dict()
        row["Entry"] = entry
        row["Theorist"] = theorist_name_log[idx]
        row["Ground Truth"] = configuration["ground_truth_name"]
        row["Mean Squared Error"] = MSE_log[idx]
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


# print the data_closed_loop frame
print(df_validation)

# MSE PLOT
sns.barplot(
    data=df_validation, x="Ground Truth", y="Mean Squared Error", hue="Theorist"
)
plt.show()


# MODEL PLOT
plot_fnc, plot_title = plot_inventory[ground_truth_name]
# get the rows of data_closed_loop with the lowest Mean Squared Error for each theorist
df_best_theories = df_validation.loc[
    df_validation.groupby(["Theorist"])["Mean Squared Error"].idxmin()
]
model_entry = df_best_theories[df_best_theories["Theorist"] == plot_theorist][
    "Entry"
].values
best_theory = full_theory_log[model_entry[0]]

# get information from the ground truth model
if ground_truth_name not in model_inventory.keys():
    raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
(metadata, data_fnc, experiment) = retrieve(ground_truth_name, kind="model:v0")
plot_fnc(best_theory)

print(best_theory.latex())
print(best_theory.__repr__())
