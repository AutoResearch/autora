# load data
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set the path to the data directory
path = './data/'
ground_truth_name = 'weber_fechner'

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

# print the list of loaded pickles
for pickle in loaded_pickles:

    configuration = pickle[0]
    MSE_log = pickle[1]
    cycle_log = pickle[2]
    repetition_log = pickle[3]
    theory_log = pickle[4]

    for idx in range(len(MSE_log)):
        row = dict()
        row["Theorist"] = configuration["theorist_name"]
        row["Experimentalist"] = configuration["experimentalist_name"]
        row["Repetition"] = repetition_log[idx]
        row["Data Collection Cycle"] = cycle_log[idx]
        row["Mean Squared Error"] = MSE_log[idx]
        df_validation = df_validation.append(row, ignore_index=True)

print(df_validation)

sns.set_theme()
rel = sns.relplot(
    data=df_validation, kind="line",
    x="Data Collection Cycle", y="Mean Squared Error",
    hue="Experimentalist", style="Experimentalist"
)
rel.fig.suptitle('Weber-Fechner Law')
# plt.title("Weber-Fechner Law")
plt.show()






