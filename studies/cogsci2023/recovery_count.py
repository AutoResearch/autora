import pandas as pd
import numpy as np
import glob
import pickle
###########################

# files = glob.glob(r'Chads priors/*.pickle')
files = glob.glob(r'BMS_Priors_299/*.pickle')
# files += glob.glob(r'data_prior_0.025/*.pickle')

num_repetitions = 15

theorists = [
    'BMS Prior Williams2023SUPERUniformNew',
    'BMS Prior Williams2023SUPERCognitivePsychologyNew',
    'BMS Prior Williams2023SUPERCognitiveScienceNew',
    'BMS Prior Williams2023SUPERMaterialsScienceNew',
    'BMS Prior Williams2023SUPERNeuroscienceNew',
    ]

gts = [
       'weber_fechner',
       'stevens_power_law',
       'exp_learning',
       'shepard_luce_choice',
       ]

repetitions = np.arange(num_repetitions)
gt_ids = np.arange(len(gts))
theorist_ids = np.arange(len(theorists))
conditions = np.array(np.meshgrid(repetitions, gt_ids, theorist_ids)).T.reshape(-1,3)

# col_names = ['Entry', 'Theorist', 'Ground Truth', 'Mean Squared Error', 'Description Length', 'Bayesian Information Criterion', 'Log Likelihood']
# data = pd.DataFrame(np.nan, index=list(np.arange(num_repetitions*len(gts)*len(theorists))), columns=col_names)

col_names = ['Prior', 'Ground Truth', 'eq']
df = pd.DataFrame(columns=col_names)
for file in files:
    model = pickle.load(open(file,'rb'))
    gt = model[0]['ground_truth_name']
    prior = model[3][0]
    eq = model[2][0]
    row = pd.Series([prior, gt, eq])
    row_df = pd.DataFrame({'Prior': [prior], 'Ground Truth': [gt], 'eq': [eq]})
    df = df.append(row_df)

print(df['Prior'].unique())
print(df['Ground Truth'].unique())

for gt in gts:
    print(f'\n{20 * "%"}\n{5 * "%"} Ground Truth: {gt}\n{20 * "%"}\n')
    df_gt = df.loc[df['Ground Truth'] == gt]
    for theorist in theorists:
        print(f'\n{10 * "%"} Prior: {theorist}\n')
        df_gt_theorist = df_gt.loc[df_gt['Prior'] == theorist]
        eqs = df_gt_theorist['eq'].tolist()
        print(len(eqs))
        for eq in eqs:
            print(eq)


# print('***************************\n\n')
# print(file)
# print(f"Ground Truth: {model[0]['ground_truth_name']}\n")
#
# print(f"Prior: {model[3][0]}\n")
#
# print("Recovered model:")
# print(model[2][0])
# print('\n')
#
# input('Press any key to continue')
