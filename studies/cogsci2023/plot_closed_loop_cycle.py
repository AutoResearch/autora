# load data
import os
import pickle
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from studies.cogsci2023.models.models import model_inventory

# set the path to the data directory
path = './data/'
ground_truth_name = 'prospect_theory'
# OPTIONS: see models.py

# TODO: - implement a better way to exclude outliers (e.g., MSE above mean + 3std for each experimentalist)
# TODO: - implement a that plots a given model against the theory (possibly the best model for each experimentalist)
# TODO: - conceive a plot where we show which data points have been collected by which experimentalist (e.g., based on Peterson et al)


MSE_cutoff = 5

# create an empty list to store the loaded pickle files
loaded_pickles = []

# iterate through all files in the data directory
for file in os.listdir(path):
  # check if the file is a pickle file
  if file.endswith('.pickle') and file.startswith(ground_truth_name):
    # open the pickle file and load the contents
    with open(os.path.join(path, file), 'rb') as f:
      pickle_data = pickle.load(f)
    # append the loaded data to the list of loaded pickles
    loaded_pickles.append(pickle_data)

df_validation = pd.DataFrame()
full_theory_log = []
entry = 0

# import the loaded pickles into data frame
for pickle in loaded_pickles:

    configuration = pickle[0]
    MSE_log = pickle[1]
    cycle_log = pickle[2]
    repetition_log = pickle[3]
    theory_log = pickle[4]
    conditions_log = pickle[5]
    observations_log = pickle[6]

    for idx in range(len(MSE_log)):

        # outlier removal
        if MSE_log[idx] > MSE_cutoff:
            continue

        row = dict()
        row["Entry"] = entry
        row["Theorist"] = configuration["theorist_name"]
        row["Experimentalist"] = configuration["experimentalist_name"]
        row["Repetition"] = repetition_log[idx]
        row["Data Collection Cycle"] = cycle_log[idx]
        row["Mean Squared Error"] = MSE_log[idx]
        full_theory_log.append(theory_log[idx])
        df_validation = df_validation.append(row, ignore_index=True)
        entry = entry + 1

# print the data frame
print(df_validation)

# plot the performance of different experimentalists as a function of cycle
sns.set_theme()
rel = sns.relplot(
    data=df_validation, kind="line",
    x="Data Collection Cycle", y="Mean Squared Error",
    hue="Experimentalist", style="Experimentalist"
)
if ground_truth_name == 'weber_fechner':
    rel.fig.suptitle('Weber-Fechner Law')
elif ground_truth_name == 'prospect_theory':
    rel.fig.suptitle('Prospect Theory')
elif ground_truth_name == 'expected_value':
    rel.fig.suptitle('Expected Value Theory')
plt.show()



# plot the best-performing theory against the ground truth

# get the rows of data with the lowest Mean Squared Error for each experimentalist
entry = df_validation.loc[df_validation.groupby(['Experimentalist'])['Mean Squared Error'].idxmin()]['Entry'].values
best_model = full_theory_log[entry[0]]
print(best_model.latex())


# get information from the ground truth model
if ground_truth_name not in model_inventory.keys():
    raise ValueError(f"Study {ground_truth_name} not found in model inventory.")
(metadata, data, experiment) = model_inventory[ground_truth_name]

if ground_truth_name == 'weber_fechner':
    X_full, y_full = data

    sns.set_style("darkgrid")
    plot_mean = 3
    min_num = 30
    S1 = X_full[:,0]
    S2 = X_full[:,1]
    PDiff = y_full
    PDiff_predicted = best_model.predict(X_full)

    plt.figure(figsize=(5, 4))
    seaborn_plot = plt.axes(projection='3d')
    print(type(seaborn_plot))
    seaborn_plot.scatter3D(S1, S2, PDiff, alpha=0.3, s=1, c='b', label='Ground Truth')
    seaborn_plot.scatter3D(S1, S2, PDiff_predicted, alpha=0.3, s=1, c='r', label='Best Model')
    seaborn_plot.set_xlabel('Stimulus 1 Intensity')
    seaborn_plot.set_ylabel('Stimulus 2 Intensity')
    seaborn_plot.set_zlabel('Perceived Difference')
    seaborn_plot.set_title('Best Model: $' + best_model.latex() + '$')
    seaborn_plot.legend()
    plt.show()







