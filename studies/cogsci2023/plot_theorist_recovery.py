from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
from studies.cogsci2023.process_theorist_recovery import process_data

path = "data_theorist/"
from_dataframe = True

if not from_dataframe:
    process_data(path)
else:
    try:
        df_validation = pd.read_csv(path+'df_validation.csv')
    except:
        with open(path + "df_validation.pickle", "rb") as f:
            df_validation = pkl.load(f)
            df = df_validation
            df.loc[df['Theorist'] == 'DARTS 3 Nodes', 'Theorist'] = 'DARTS'
            df.loc[df['Ground Truth'] == 'weber_fechner', 'Ground Truth'] = 'Weber-Fechner Law'
            df.loc[df['Ground Truth'] == 'stevens_power_law', 'Ground Truth'] = 'Steven’s Power Law'
            df.loc[df['Ground Truth'] == 'exp_learning', 'Ground Truth'] = 'Exponential Learning'
            df.loc[
                df['Ground Truth'] == 'shepard_luce_choice', 'Ground Truth'] = 'Luce-Choice-Ratio'
            df.loc[df['Ground Truth'] == 'task_switching', 'Ground Truth'] = 'Task Switching'
            df.loc[df['Ground Truth'] == 'stroop_model', 'Ground Truth'] = 'Stroop Model'
            df.loc[
                df['Ground Truth'] == 'evc_congruency', 'Ground Truth'] = 'Incentivised Attention'
            df.loc[
                df['Ground Truth'] == 'evc_demand_selection', 'Ground Truth'] = 'Demand Selection'
            df.loc[df['Ground Truth'] == 'evc_coged', 'Ground Truth'] = 'Effort Discounting'
            df.loc[
                df['Ground Truth'] == 'expected_value', 'Ground Truth'] = 'Expected Utility Theory'
            df.loc[df['Ground Truth'] == 'prospect_theory', 'Ground Truth'] = 'Prospect Theory'
            df.loc[df['Ground Truth'] == 'tva', 'Ground Truth'] = 'Theory of Visual Attention'
            df_validation = df
    # with open(path + "full_theory_log.pickle", "rb") as f:
    #     theory_logs = pkl.load(f)

# df_2 = pd.read_csv('data_prior/df_validation.csv')
# print(df_2['Theorist'].unique())
# df_2 = df_2.loc[df_2['Theorist'] == 'BMS Uniform']
# df_validation = df_validation.append(df_2)

# print(theory_logs['Theorist'].unique())
print(df_validation['Theorist'].unique())
print(df_validation.columns)
print(df_validation['Ground Truth'].unique())
print(df_validation['Entry'])

GT_order = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']

study = path[5:-1]
measures = ['Mean Squared Error', 'Bayesian Information Criterion', 'Description Length']
types = ['Average', 'Best']
if path == 'data_theorist/' or path == 'data_prior/':
    theorist_order = df_validation['Theorist'].unique()[:-1]
elif path == 'data_paramter/':
    theorist_order = ['BMS Parameter['+str(i+1)+']' for i in range(8)]
elif path == 'data+fixed_root/':
    theorist_order = ['BMS', 'BMS Fixed Root', 'Regression']

theorist_order = ['Regression', 'BMS Parameter[5]', 'BMS Fixed Root Parameter[5]',
                  'BMS Parameter[10]', 'BMS Fixed Root Parameter[10]']

for measure in measures:
    for type in types:
        if type == 'Best':
            df = df_validation.loc[df_validation.groupby(["Theorist", "Ground Truth"])[measure].idxmin()]
            df = df.drop_duplicates(subset=['Ground Truth', 'Theorist'])

        else:
            df = df_validation
        figure, axis = plt.subplots(3, 4)
        for i, ground_truth_name in enumerate(GT_order):
            if ground_truth_name in df['Ground Truth'].unique():
                x = int(i/4)
                y = i % 4
                g = sns.barplot(ax=axis[x, y], data=df.loc[df['Ground Truth'] == ground_truth_name],
                                x='Theorist', y=measure,
                                order=theorist_order,
                                errorbar=('se'))
                g.set_xlabel(ground_truth_name)
                g.set_ylabel("")
                g.set_xticklabels("")
        print(type + ' for ' + ground_truth_name + ':\n' + str(
            df.loc[df['Ground Truth'] == ground_truth_name][['Theorist', 'Log Likelihood']]))

        plt.subplots_adjust(wspace=0.3)
        figure.supxlabel('Theorist ' + type + ' Model')
        figure.supylabel(measure)
        legend_elements=[]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if study == 'theorist' and measure != 'Mean Squared Error':
            colors = [colors[i] for i in [0, 2, 3, 4]]
            order = [theorist_order[i] for i in [0, 2, 3, 4]]
            print('EXCEPTION')
        else:
            order = theorist_order
        for i, theorist_name in enumerate(order):
            legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
        figure.legend(handles=legend_elements)
        plt.show()



if __name__ == '__main__':
    ...
