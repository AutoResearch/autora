# load data_closed_loop
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from autora.model import retrieve
from studies.cogsci2023.models.models import model_inventory, plot_inventory

# set the path to the data_closed_loop directory
path = "data_closed_loop/"
ground_truth_name = "prospect_theory"  # OPTIONS: see models.py
experimentalist_name = "random"  # for plotting

plot_performance = True
plot_model = False
plot_tsne = False

# create an empty list to store the loaded pickle files
loaded_pickles = []

# iterate through all files in the data_closed_loop directory
for file in os.listdir(path):
    # check if the file is a pickle file
    if file.endswith(".pickle") and file.startswith(ground_truth_name):
        # open the pickle file and load the contents
        with open(os.path.join(path, file), "rb") as f:
            print(f"Loading {file}")
            pickle_data = pickle.load(f)
        # append the loaded data_closed_loop to the list of loaded pickles
        loaded_pickles.append(pickle_data)

df_validation = pd.DataFrame()
full_theory_log = []
full_conditions_log = []
full_observations_log = []
entry = 0

# import the loaded pickles into data_closed_loop frame
for pickle in loaded_pickles:

    configuration = pickle[0]
    MSE_log = pickle[1]
    cycle_log = pickle[2]
    repetition_log = pickle[3]
    theory_log = pickle[4]
    conditions_log = pickle[5]
    observations_log = pickle[6]
    experimentalist_log = pickle[7]

    for idx in range(len(MSE_log)):

        # check if MSE is nan or infinite
        if np.isnan(MSE_log[idx]) or np.isinf(MSE_log[idx]):
            continue

        row = dict()
        row["Entry"] = entry
        row["Theorist"] = configuration["theorist_name"]
        row["Experimentalist"] = experimentalist_log[idx]
        row["Repetition"] = repetition_log[idx]
        row["Data Collection Cycle"] = cycle_log[idx]
        row["Mean Squared Error"] = MSE_log[idx]
        full_theory_log.append(theory_log[idx])
        full_conditions_log.append(conditions_log[idx])
        full_observations_log.append(observations_log[idx])
        df_validation = df_validation.append(row, ignore_index=True)

        entry = entry + 1

# remove MSE outliers
experimentalists = df_validation["Experimentalist"].unique()
cycles = df_validation["Data Collection Cycle"].unique()
for experimenalist in experimentalists:
    for cycle in cycles:
        # compute the mean MSE for each experimentalist in the data_closed_loop frame
        mean_MSE = df_validation[
            (df_validation["Experimentalist"] == experimenalist)
            & (df_validation["Data Collection Cycle"] == cycle)
        ]["Mean Squared Error"].mean()
        std_MSE = df_validation[
            (df_validation["Experimentalist"] == experimenalist)
            & (df_validation["Data Collection Cycle"] == cycle)
        ]["Mean Squared Error"].std()
        # remove all rows with MSE above the mean + 3 standard deviations
        df_validation = df_validation.drop(
            df_validation[
                (df_validation["Experimentalist"] == experimenalist)
                & (df_validation["Data Collection Cycle"] == cycle)
                & (df_validation["Mean Squared Error"] > mean_MSE + 3 * std_MSE)
            ].index
        )


# print the data_closed_loop frame
print(df_validation)
df_validation.to_csv(path + "/" + ground_truth_name + "_MSE.csv")

# copy data set
df_entropy = df_validation.copy()
df_entropy = df_entropy.drop(
    df_entropy[df_entropy["Data Collection Cycle"] != configuration["num_cycles"]].index
)
entropy_log = []
for index, row in df_entropy.iterrows():
    # compute entropy of the corresponding observations
    entry = row["Entry"]
    y = np.array(full_observations_log[entry])
    # compute entropy of y
    entropy = -np.sum(y * np.log(y))
    entropy_log.append(entropy)
df_entropy["Entropy"] = entropy_log
df_validation.to_csv(path + "/" + ground_truth_name + "_entropy.csv")


# CYCLE PLOT
if plot_performance:
    plot_fnc, plot_title = plot_inventory[ground_truth_name]

    # plot the performance of different experimentalists as a function of cycle
    sns.set_theme()
    rel = sns.relplot(
        data=df_validation,
        kind="line",
        x="Data Collection Cycle",
        y="Mean Squared Error",
        hue="Experimentalist",
        style="Experimentalist",
    )
    rel.fig.suptitle(plot_title)
    plt.show()


# MODEL PLOT

if plot_model:
    # get the rows of data_closed_loop with the lowest Mean Squared Error for each experimentalist
    df_final_cycle = df_validation[
        df_validation["Data Collection Cycle"]
        == df_validation["Data Collection Cycle"].max()
    ]
    df_best_theories = df_final_cycle.loc[
        df_final_cycle.groupby(["Experimentalist"])["Mean Squared Error"].idxmin()
    ]
    model_entry = df_best_theories[
        df_best_theories["Experimentalist"] == experimentalist_name
    ]["Entry"].values
    best_theory = full_theory_log[model_entry[0]]

    # get information from the ground truth model
    if ground_truth_name not in model_inventory.keys():
        raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
    (metadata, data_fnc, experiment) = retrieve(ground_truth_name, kind="model:v0")
    plot_fnc(best_theory)


# PLOT THE DATA COLLECTED BY THE BEST-PERFORMING EXPERIMENTALIST

if plot_tsne:
    # collect ground truth data_closed_loop
    full_data_label = "full data set"
    X, y = data_fnc(metadata)
    labels = list()
    cycle_count = list()
    # populate the labels list with "full data_closed_loop set" for each data_closed_loop point in X
    for i in range(len(X)):
        labels.append(full_data_label)
        cycle_count.append(configuration["num_cycles"])

    # collect data_closed_loop collected by the best-performing experimentalist
    for experimenalist in experimentalists:
        entry = df_best_theories[df_best_theories["Experimentalist"] == experimenalist][
            "Entry"
        ].values
        conditions = full_conditions_log[entry[0]]
        X = np.vstack((X, conditions))
        for i in range(len(conditions)):
            cycle_count.append(np.floor_divide(i, configuration["samples_per_cycle"]))
            labels.append(experimenalist)

    # T-SNE Transformation
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(X)

    # create final data_closed_loop frame
    df = pd.DataFrame()
    df["Labels"] = labels
    df["Component 1"] = z[:, 0]
    df["Component 2"] = z[:, 1]
    for col in range(X.shape[1]):
        df[f"Condition {col}"] = X[:, col]
    df["Cycle"] = cycle_count

    # create separate df for full data set
    full_data_only = df[df["Labels"] == "full data set"]
    full_data_only["y"] = y

    # drop conditions with the seed cycle and full data set
    df.drop(df[df["Cycle"] == 0].index, inplace=True)
    df.drop(df[df["Labels"] == full_data_label].index, inplace=True)

    # T-SNE plot for full data
    custom_palette = sns.color_palette("Greys", len(y.tolist()))
    sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue=full_data_only.y,
        palette=custom_palette,
        data=full_data_only,
        linewidth=0,
        # s = 10,
        legend=False,
    )
    # plt.show()

    # x_list = full_data_only["Component 1"]
    # y_list = full_data_only["Component 2"]
    # z_list = full_data_only["y"]
    #
    # from scipy.interpolate import interp2d
    # f = interp2d(x_list, y_list, z_list, kind="linear")
    #
    # x_coords = np.arange(min(x_list), max(x_list) + 1, step=0.01)
    # y_coords = np.arange(min(y_list), max(y_list) + 1, step=0.01)
    # z = f(x_coords, y_coords)
    # plt.pcolormesh(x_coords, y_coords, z, vmin=0, vmax=1, cmap="Greys")
    # plt.show()

    # T-SNE Plot for experimentalists
    custom_palette = sns.color_palette("deep", len(set(labels)))
    # custom_palette[0] = (0.5, 0.5, 0.5)
    sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue=df.Labels.tolist(),
        palette=custom_palette,
        s=10,
        data=df,
    ).set(
        title="T-SNE Projection of Probed Experimental Conditions\n(" + plot_title + ")"
    )
    plt.show()

    sns.histplot(data=full_data_only, x="y", bins=100)
    plt.show()

# plot best theory
if plot_model:
    print(best_theory.model_.latex())
