# load data_closed_loop
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
from studies.cogsci2023.models.models import model_inventory, plot_inventory

# set the path to the data_closed_loop directory
path = 'data_closed_loop/'
ground_truth_name = 'prospect_theory' # OPTIONS: see models.py
experimentalist_name = 'Random' # for plotting
max_cycle_mse = 50 # None
max_cycle_tnse = 20 # None

plot_performance = True
plot_model = False
plot_tsne = False
print_model = False
legend_type = "auto" # False, "auto"
figure_dim = (5.5, 5.5)

experimentalist_labels = dict()
experimentalist_labels['dissimilarity'] = 'Novelty'
experimentalist_labels['falsification'] = 'Falsification'
experimentalist_labels['least confident'] = 'Least Confident'
experimentalist_labels['model disagreement'] = 'Model Disagreement'
experimentalist_labels['random'] = 'Random'
experimentalist_labels['popper'] = 'Popper'

experimentalist_order = list()
experimentalist_order.append(experimentalist_labels['random'])
experimentalist_order.append(experimentalist_labels['dissimilarity'])
experimentalist_order.append(experimentalist_labels['least confident'])
experimentalist_order.append(experimentalist_labels['model disagreement'])
experimentalist_order.append(experimentalist_labels['falsification'])
experimentalist_order.append(experimentalist_labels['popper'])


if legend_type == "auto":
    figure_dim = (8.5, 5.5)


# create an empty list to store the loaded pickle files
loaded_pickles = []

# iterate through all files in the data_closed_loop directory
for file in os.listdir(path):
    # check if the file is a pickle file
    if file.endswith('.pickle') and file.startswith(ground_truth_name):
        # open the pickle file and load the contents
        with open(os.path.join(path, file), 'rb') as f:
            print(f'Loading {file}')
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

        if max_cycle_mse is not None:
            if cycle_log[idx] > max_cycle_mse:
                continue

        row = dict()
        row["Entry"] = entry
        row["Theorist"] = configuration["theorist_name"]
        row["Experimentalist"] = experimentalist_labels[experimentalist_log[idx]]
        row["Repetition"] = repetition_log[idx]
        row["Data Collection Cycle"] = cycle_log[idx]
        row["Mean Squared Error"] = MSE_log[idx]
        full_theory_log.append(theory_log[idx])
        full_conditions_log.append(conditions_log[idx])
        full_observations_log.append(observations_log[idx])
        df_validation = df_validation.append(row, ignore_index=True)
        # df_validation = pd.concat([df_validation, pd.DataFrame.from_records([row])])

        entry = entry + 1

# remove MSE outliers
experimentalists = df_validation["Experimentalist"].unique()
cycles = df_validation["Data Collection Cycle"].unique()
for experimenalist in experimentalists:
    for cycle in cycles:
        # compute the mean MSE for each experimentalist in the data_closed_loop frame
        mean_MSE = df_validation[(df_validation["Experimentalist"] == experimenalist) &
                                 (df_validation["Data Collection Cycle"] == cycle)][
            "Mean Squared Error"].mean()
        std_MSE = df_validation[(df_validation["Experimentalist"] == experimenalist) &
                                (df_validation["Data Collection Cycle"] == cycle)][
            "Mean Squared Error"].std()
        # remove all rows with MSE above the mean + 3 standard deviations
        df_validation = df_validation.drop(
            df_validation[(df_validation["Experimentalist"] == experimenalist) &
                          (df_validation["Data Collection Cycle"] == cycle) &
                          (df_validation["Mean Squared Error"] > mean_MSE + 3 * std_MSE)].index)


# print the data_closed_loop frame
print(df_validation)
df_final_mse = df_validation.copy()
df_final_mse = df_final_mse.drop(df_final_mse[df_final_mse["Data Collection Cycle"] != max_cycle_mse].index)
df_final_mse.to_csv(path + "/" + ground_truth_name + "_MSE.csv")

# copy data set
df_entropy = df_validation.copy()
df_entropy = df_entropy.drop(df_entropy[df_entropy["Data Collection Cycle"] != max_cycle_mse].index)
entropy_log = []
for index, row in df_entropy.iterrows():
    # compute entropy of the corresponding observations
    entry = row["Entry"]
    y = np.array(full_observations_log[entry])
    # compute entropy of y
    entropy = -np.sum(y * np.log(y))
    entropy_log.append(entropy)
df_entropy["Entropy"] = entropy_log
df_entropy.to_csv(path + "/" + ground_truth_name + "_entropy.csv")

# variance of the observations
df_variance = df_validation.copy()
df_variance = df_variance.drop(df_variance[df_variance["Data Collection Cycle"] != max_cycle_mse].index)
variance_log = []
for index, row in df_variance.iterrows():
    # compute variance of the corresponding observations
    entry = row["Entry"]
    y = np.array(full_observations_log[entry])
    # compute variance of y
    variance = np.var(y)
    variance_log.append(variance)
df_variance["Variance"] = variance_log
df_variance.to_csv(path + "/" + ground_truth_name + "_variance.csv")

sns.set(rc={'figure.figsize':figure_dim})
sns.set(font_scale=1.3)
plot_fnc, plot_title = plot_inventory[ground_truth_name]

# CYCLE PLOT
if plot_performance:
    # plot the performance of different experimentalists as a function of cycle
    # sns.set_theme()
    rel = sns.relplot(
        data=df_validation, kind="line",
        x="Data Collection Cycle", y="Mean Squared Error",
        hue="Experimentalist", style="Experimentalist", legend=legend_type, # False
        hue_order=experimentalist_order,
    )
    rel.fig.subplots_adjust(top=.90)
    rel.fig.subplots_adjust(bottom=0.2)
    rel.fig.suptitle(plot_title)
    rel.fig.set_size_inches(figure_dim[0],figure_dim[1]-2)
    # leg = rel._legend
    # leg.set_bbox_to_anchor([0.5, 0.5])  # coordinates of lower left of bounding box
    # leg._loc = 2
    plt.show()


# MODEL PLOT

if plot_model:
    # get the rows of data_closed_loop with the lowest Mean Squared Error for each experimentalist
    df_final_cycle = df_validation[df_validation["Data Collection Cycle"] == df_validation["Data Collection Cycle"].max()]
    df_best_theories = df_final_cycle.loc[df_final_cycle.groupby(['Experimentalist'])['Mean Squared Error'].idxmin()]
    model_entry = df_best_theories[df_best_theories["Experimentalist"] == experimentalist_name]['Entry'].values
    best_theory = full_theory_log[model_entry[0]]

    # get information from the ground truth model
    if ground_truth_name not in model_inventory.keys():
        raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
    (metadata, data_fnc, experiment) = model_inventory[ground_truth_name]
    plot_fnc(best_theory)


# PLOT THE DATA COLLECTED BY THE BEST-PERFORMING EXPERIMENTALIST

if plot_tsne:
    df_tsne_cycle = df_validation[
        df_validation["Data Collection Cycle"] == max_cycle_tnse]
    df_best_theories = df_tsne_cycle.loc[
        df_tsne_cycle.groupby(['Experimentalist'])['Mean Squared Error'].idxmin()]

    # collect ground truth data_closed_loop
    full_data_label = "full data set"
    X, y = data_fnc(metadata)
    # T-SNE Transformation
    tsne = TSNE(n_components=2, verbose=1, random_state=111)
    z = tsne.fit_transform(X)

    full_data_only = pd.DataFrame()
    full_data_only["Component 1"] = z[:, 0]
    full_data_only["Component 2"] = z[:, 1]
    full_data_only["y"] = y

    labels = list()
    cycle_count = list()
    comp1 = list()
    comp2 = list()
    # populate the labels list with "full data_closed_loop set" for each data_closed_loop point in X
    # for i in range(len(X)):
    #     labels.append(full_data_label)
    #     cycle_count.append(configuration["num_cycles"])

    # collect data_closed_loop collected by the best-performing experimentalist
    for experimenalist in experimentalists:
        entry = df_best_theories[(df_best_theories["Experimentalist"] == experimenalist)]['Entry'].values
        conditions = full_conditions_log[entry[0]]
        # X = np.vstack((X, conditions))
        for i in range(len(conditions)):
            condition = conditions[i]
            # find index of row in X that matches condition
            index = np.where((X == condition).all(axis=1))[0][0]
            comp1.append(z[index, 0])
            comp2.append(z[index, 1])
            cycle_count.append(np.floor_divide(i, configuration["samples_per_cycle"]))
            labels.append(experimenalist)



    # create final data_closed_loop frame
    df = pd.DataFrame()
    df["Labels"] = labels
    df["Component 1"] = comp1
    df["Component 2"] = comp2
    df["Cycle"] = cycle_count
    # for col in range(X.shape[1]):
    #     df[f"Condition {col}"] = X[:,col]

    #
    # # create separate df for full data set
    # full_data_only = df[df["Labels"] == 'full data set']
    # full_data_only["y"] = y
    #
    # # drop conditions with the seed cycle and full data set
    # df.drop(df[df['Cycle'] == 0].index, inplace=True)
    # df.drop(df[df['Labels'] == full_data_label].index, inplace=True)

    # T-SNE plot for full data
    custom_palette = sns.color_palette("Greys", len(y.tolist()))
    g = sns.scatterplot(x="Component 1", y="Component 2", hue=full_data_only.y,
                    palette=custom_palette,
                    data=full_data_only,
                    linewidth = 0,
                    # s = 10,
                    legend=False)

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
    sns.set(font_scale=1.3)
    pl = sns.scatterplot(x="Component 1", y="Component 2", hue=df.Labels.tolist(),
                    palette=custom_palette,
                    hue_order=experimentalist_order,
                    # s = 10,
                    alpha=0.5,
                    legend=legend_type,
                    data=df).set(title="T-SNE Projection of Probed Experimental Conditions\n(" + plot_title + ")")
    # pl.set_xlabel("X Label",fontsize=30)
    plt.show()

    sns.histplot(data=full_data_only, x="y", bins=100)
    plt.show()

# output best theory
if print_model:
    if hasattr(best_theory, "model_"):
        print(best_theory.model_.latex())





