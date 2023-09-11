# load data_closed_loop
import os
import pickle as pkl
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

from studies.cogsci2023.models.models import model_inventory, plot_inventory


def change_gt_names(df):
    # df.loc[df['Theorist'] == 'DARTS 3 Nodes', 'Theorist'] = 'DARTS'
    df.loc[df['Ground Truth'] == 'weber_fechner', 'Ground Truth'] = 'Weber-Fechner Law'
    df.loc[df['Ground Truth'] == 'stevens_power_law', 'Ground Truth'] = 'Steven’s Power Law'
    df.loc[df['Ground Truth'] == 'exp_learning', 'Ground Truth'] = 'Exponential Learning'
    df.loc[df['Ground Truth'] == 'shepard_luce_choice', 'Ground Truth'] = 'Luce-Choice-Ratio'
    df.loc[df['Ground Truth'] == 'task_switching', 'Ground Truth'] = 'Task Switching'
    df.loc[df['Ground Truth'] == 'stroop_model', 'Ground Truth'] = 'Stroop Model'
    df.loc[df['Ground Truth'] == 'evc_congruency', 'Ground Truth'] = 'Incentivised Attention'
    df.loc[df['Ground Truth'] == 'evc_demand_selection', 'Ground Truth'] = 'Demand Selection'
    df.loc[df['Ground Truth'] == 'evc_coged', 'Ground Truth'] = 'Effort Discounting'
    df.loc[df['Ground Truth'] == 'expected_value', 'Ground Truth'] = 'Expected Utility Theory'
    df.loc[df['Ground Truth'] == 'prospect_theory', 'Ground Truth'] = 'Prospect Theory'
    df.loc[df['Ground Truth'] == 'tva', 'Ground Truth'] = 'Theory of Visual Attention'
    return df


def process_data(path="data_theorist/", retrain=False):
    df_validation = pd.DataFrame()
    theory_logs = dict()
    full_theory_log = []
    entry = 0
    trained = []
    for gt_model in os.listdir('models/'):
        if 'df_validation_'+gt_model[:-3]+'.pickle' in os.listdir(path):
            trained.append(gt_model)
    start = datetime.now()
    for gt_model in os.listdir('models/'):
        stop = datetime.now()
        print((stop-start).total_seconds())
        start = stop
        if gt_model == '__init__.py' or gt_model == 'models.py' or not gt_model.endswith('.py'):
            continue
        found = False
        for file in os.listdir(path):
            # check if the file is a pickle file
            if file.endswith(".pickle") and file.startswith(gt_model[:-3]):
                found = True
        if found is False:
            continue
        if gt_model.startswith('df_validation') or (gt_model in trained and not retrain):
            print('skipped because already trained')
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
            for idx in range(len(MSE_log)):

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
                full_theory_log.append((theory_log[idx], MSE_log[idx]))
                df_validation = df_validation.append(row, ignore_index=True)
                entry = entry + 1
        theory_logs.update({ground_truth_name: full_theory_log})
        print(f'columns:{df_validation.columns}')
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
        file_name = path + "df_validation_"+ground_truth_name+".pickle"
        with open(file_name, "wb") as f:
            pkl.dump(df_validation, f)
        # save and load pickle file
        file_name = path + "full_theory_log_"+ground_truth_name+".pickle"
        with open(file_name, "wb") as g:
            pkl.dump(theory_logs, g)
    df_validation = pd.DataFrame()
    for gt_model in os.listdir('models/'):
        df_file = 'df_validation_' + gt_model[:-3] + '.pickle'
        if df_file in os.listdir(path):
            with open(path + df_file, "rb") as f:
                df = pkl.load(f)
            df_validation = df_validation.append(df)

    df_validation = change_gt_names(df_validation)
    df_validation.to_csv(path + "df_validation.csv", index=False)

    theory_log = pd.DataFrame(theory_logs)
    # for gt_model in os.listdir('models/'):
    #     df_file = path + "full_theory_log_"+gt_model[:-3]+".pickle"
    #     if df_file in os.listdir(path):
    #         with open(path + df_file, "rb") as f:
    #             df = pkl.load(f)
    #         theory_log = theory_log.append(df)
    #         print(df)

    # theory_log = change_gt_names(theory_log)
    theory_log.to_csv(path + "theory_log.csv", index=False)


if __name__ == '__main__':
    process_data('prior_0.25/')

# with open(os.path.join(path, 'evc_coged_8.pickle'), "rb") as f:
#     pickle_data = pkl.load(f)
#
# print(pickle_data[2][-3].model_)
# print(pickle_data[2][-3].model_.par_values)
# print(pickle_data[1][-3])
