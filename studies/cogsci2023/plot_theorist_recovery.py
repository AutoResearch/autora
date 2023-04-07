# load data_closed_loop
import os
import pickle as pkl
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from studies.cogsci2023.models.models import model_inventory, plot_inventory
from studies.cogsci2023.process_theorist_recovery import process_data

path = "data_parameter/"
from_dataframe = True

# with open(os.path.join(path, 'evc_coged_8.pickle'), "rb") as f:
#     pickle_data = pkl.load(f)
#
# print(pickle_data[2][-3].model_)
# print(pickle_data[2][-3].model_.par_values)
# print(pickle_data[1][-3])

if not from_dataframe:
    process_data(path)
else:
    with open(path + "df_validation.pickle", "rb") as f:
        df_validation = pkl.load(f)
    # with open(path + "full_theory_log.pickle", "rb") as f:
    #     theory_logs = pkl.load(f)

# print(theory_logs['Theorist'].unique())
print(df_validation['Theorist'].unique())
print(df_validation.columns)

df_validation.loc[df_validation['Theorist'] == 'DARTS 3 Nodes', 'Theorist'] = 'DARTS'
df_validation.loc[df_validation['Ground Truth'] == 'weber_fechner', 'Ground Truth'] = 'Weber-Fechner Law'
df_validation.loc[df_validation['Ground Truth'] == 'stevens_power_law', 'Ground Truth'] = 'Steven’s Power Law'
df_validation.loc[df_validation['Ground Truth'] == 'exp_learning', 'Ground Truth'] = 'Exponential Learning'
df_validation.loc[df_validation['Ground Truth'] == 'shepard_luce_choice', 'Ground Truth'] = 'Luce-Choice-Ratio'
df_validation.loc[df_validation['Ground Truth'] == 'task_switching', 'Ground Truth'] = 'Task Switching'
df_validation.loc[df_validation['Ground Truth'] == 'stroop_model', 'Ground Truth'] = 'Stroop Model'
df_validation.loc[df_validation['Ground Truth'] == 'evc_congruency', 'Ground Truth'] = 'Incentivised Attention'
df_validation.loc[df_validation['Ground Truth'] == 'evc_demand_selection', 'Ground Truth'] = 'Demand Selection'
df_validation.loc[df_validation['Ground Truth'] == 'evc_coged', 'Ground Truth'] = 'Effort Discounting'
df_validation.loc[df_validation['Ground Truth'] == 'expected_value', 'Ground Truth'] = 'Expected Utility Theory'
df_validation.loc[df_validation['Ground Truth'] == 'prospect_theory', 'Ground Truth'] = 'Prospect Theory'
df_validation.loc[df_validation['Ground Truth'] == 'tva', 'Ground Truth'] = 'Theory of Visual Attention'

df_best_mse = df_validation.loc[df_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]
df_best_bic = df_validation.loc[df_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]
df_best_dl = df_validation.loc[df_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_best_bic.loc[df_best_bic['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Bayesian Information Criterion',
                        order=['BMS Prior', 'BMS Williams2023Psychophysics',
 'BMS Williams2023PsychophysicsUpWeighted',
 'BMS Williams2023CognitivePsychology',
 'BMS Williams2023CognitivePsychologyUpWeighted',
 'BMS Williams2023BehavioralEconomics',
 'BMS Williams2023BehavioralEconomicsUpWeighted'],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Bayesian Information Criterion')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS Prior', 'BMS Williams2023Psychophysics',
 'BMS Williams2023PsychophysicsUpWeighted',
 'BMS Williams2023CognitivePsychology',
 'BMS Williams2023CognitivePsychologyUpWeighted',
 'BMS Williams2023BehavioralEconomics',
 'BMS Williams2023BehavioralEconomicsUpWeighted']):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_validation.loc[df_validation['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Mean Squared Error',
                        order=['BMS', 'BMS Fixed Root'],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Average Model')
figure.supylabel('Mean Squared Error')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS', 'BMS Fixed Root']):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_best_mse['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_best_mse.loc[df_best_mse['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Mean Squared Error',
                        order=['BMS', 'BMS Fixed Root'],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Mean Squared Error')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS', 'BMS Fixed Root']):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_validation.loc[df_validation['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Bayesian Information Criterion',
                        order=['BMS', 'BMS Fixed Root'],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Average Model')
figure.supylabel('Bayesian Information Criterion')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS', 'BMS Fixed Root']):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()


df_param_validation = df_validation.loc[df_validation['Theorist'] > 'BMS Parameter[0]']
df_param_validation = df_param_validation.loc[df_validation['Theorist'] < 'BMS Parameter[9]']

df_best_mse = df_param_validation.loc[df_param_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]
df_best_bic = df_param_validation.loc[df_param_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]
df_best_dl = df_param_validation.loc[df_param_validation.groupby(["Theorist", "Ground Truth"])["Mean Squared Error"].idxmin()]

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_param_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_param_validation.loc[df_validation['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Mean Squared Error',
                        order=['BMS Parameter['+str(t)+']' for t in range(1, 8)],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Average Model')
figure.supylabel('Mean Squared Error')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS Parameter['+str(t)+']' for t in range(1, 9)]):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_param_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_best_mse.loc[df_best_mse['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Mean Squared Error',
                        order=['BMS Parameter['+str(t)+']' for t in range(1, 9)],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Mean Squared Error')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS Parameter['+str(t)+']' for t in range(1, 9)]):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_param_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_best_bic.loc[df_best_bic['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Bayesian Information Criterion',
                        order=['BMS Parameter['+str(t)+']' for t in range(1, 9)],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Bayesian Information Criterion')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS Parameter['+str(t)+']' for t in range(1, 9)]):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
# sns.barplot(ax=axis[0, 0], data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
#                 hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
GT_list = ['Weber-Fechner Law', 'Steven’s Power Law', 'Exponential Learning', 'Luce-Choice-Ratio',
           'Task Switching', 'Stroop Model', 'Incentivised Attention', 'Theory of Visual Attention',
           'Demand Selection','Effort Discounting', 'Expected Utility Theory', 'Prospect Theory']
for i, ground_truth_name in enumerate(GT_list):
    if ground_truth_name in df_param_validation['Ground Truth'].unique():
        x = int(i/4)
        y = i % 4
        g = sns.barplot(ax=axis[x, y], data=df_best_dl.loc[df_best_dl['Ground Truth'] == ground_truth_name],
                        x='Theorist', y='Description Length',
                        order=['BMS Parameter['+str(t)+']' for t in range(1, 9)],
                        errorbar=('se'))
        g.set_xlabel(ground_truth_name)
        g.set_ylabel("")
        g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Minimum Description Length')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
new_colors = [colors[0], colors[2], colors[3], colors[4]]
for i, theorist_name in enumerate(['BMS Parameter['+str(t)+']' for t in range(1, 9)]):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, ground_truth_name in enumerate(GT_list):
    x = int(i/4)
    y = i % 4
    g = sns.barplot(ax=axis[x, y], data=df_best_mse.loc[df_best_mse['Ground Truth'] == ground_truth_name],
                x='Theorist', y='Mean Squared Error', order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'],
                errorbar=('se'))
    g.set_xlabel(ground_truth_name)
    g.set_ylabel("")
    g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Mean Squared Error')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i, theorist_name in enumerate(['Regression', 'MLP', 'DARTS', 'BSR', 'BMS']):
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, ground_truth_name in enumerate(GT_list):
    x = int(i/4)
    y = i % 4
    g = sns.barplot(ax=axis[x, y], data=df_validation.loc[df_validation['Ground Truth'] == ground_truth_name],
                x='Theorist', y='Bayesian Information Criterion', order=['Regression', 'DARTS', 'BSR', 'BMS'], palette=new_colors,
                errorbar=('se'))
    g.set_xlabel(ground_truth_name)
    g.set_ylabel("")
    g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Average Model')
figure.supylabel('Bayesian Information Criterion')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i, theorist_name in enumerate(['Regression', 'MLP', 'DARTS', 'BSR', 'BMS']):
    if theorist_name == 'MLP':
        continue
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, ground_truth_name in enumerate(GT_list):
    x = int(i/4)
    y = i % 4
    g = sns.barplot(ax=axis[x, y], data=df_best_bic.loc[df_best_bic['Ground Truth'] == ground_truth_name],
                x='Theorist', y='Bayesian Information Criterion', order=['Regression', 'DARTS', 'BSR', 'BMS'], palette=new_colors,
                errorbar=('se'))
    g.set_xlabel(ground_truth_name)
    g.set_ylabel("")
    g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Bayesian Information Criterion')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i, theorist_name in enumerate(['Regression', 'MLP', 'DARTS', 'BSR', 'BMS']):
    if theorist_name == 'MLP':
        continue
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, ground_truth_name in enumerate(GT_list):
    x = int(i/4)
    y = i % 4
    g = sns.barplot(ax=axis[x, y], data=df_validation.loc[df_validation['Ground Truth'] == ground_truth_name], palette=new_colors,
                x='Theorist', y='Description Length', order=['Regression', 'DARTS', 'BSR', 'BMS'],
                errorbar=('se'))
    g.set_xlabel(ground_truth_name)
    g.set_ylabel("")
    g.set_xticklabels("")
    plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Average Model')
figure.supylabel('Minimum Description Length')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i, theorist_name in enumerate(['Regression', 'MLP', 'DARTS', 'BSR', 'BMS']):
    if theorist_name == 'MLP':
        continue
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
figure.legend(handles=legend_elements)
plt.show()

figure, axis = plt.subplots(3, 4)
for i, theorist_name in enumerate(['Regression', 'MLP', 'DARTS', 'BSR', 'BMS']):
    if theorist_name == 'MLP':
        continue
    legend_elements.append(Patch(facecolor=colors[i], label=theorist_name))
for i, ground_truth_name in enumerate(GT_list):
    x = int(i/4)
    y = i % 4
    g = sns.barplot(ax=axis[x, y], data=df_best_dl.loc[df_best_dl['Ground Truth'] == ground_truth_name], palette=new_colors,
                x='Theorist', y='Description Length', order=['Regression', 'DARTS', 'BSR', 'BMS'],
                errorbar=('se'))
    g.set_xlabel(ground_truth_name)
    g.set_ylabel("")
    g.set_xticklabels("")
plt.subplots_adjust(wspace=0.3)
figure.supxlabel('Theorist Best Model')
figure.supylabel('Minimum Description Length')
legend_elements=[]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

figure.legend(handles=legend_elements)
plt.show()

# MSE mean Plot
g = sns.barplot(data=df_validation, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
                hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.set_ylim(0, 0.1)
plt.tight_layout()
plt.show()

# df_best_mse.loc[df_best_mse['Theorist'] == 'BMS'][]

# MSE min Plot
g = sns.barplot(data=df_best_mse, x='Ground Truth', y='Mean Squared Error', hue='Theorist',
                hue_order=['Regression', 'MLP', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.set_ylim(0, 0.1)
plt.tight_layout()
plt.show()

# Bayesian Information Criterion mean Plot
g = sns.barplot(data=df_validation, x='Ground Truth', y='Bayesian Information Criterion', hue='Theorist',
                hue_order=['Regression', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
# g.set_ylim(-20, 30)
plt.tight_layout()
plt.show()

# Bayesian Information Criterion min Plot
g = sns.barplot(data=df_best_bic, x='Ground Truth', y='Bayesian Information Criterion', hue='Theorist',
                hue_order=['Regression', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
# g.set_ylim(-20, 30)
plt.tight_layout()
plt.show()

# Description Length mean Plot
g = sns.barplot(data=df_validation, x='Ground Truth', y='Description Length', hue='Theorist',
                hue_order=['Regression', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
# g.set_ylim(-10, 10)
plt.tight_layout()
plt.show()

# Description Length min Plot
g = sns.barplot(data=df_best_dl, x='Ground Truth', y='Description Length', hue='Theorist',
                hue_order=['Regression', 'DARTS', 'BSR', 'BMS'], errorbar=('se'))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
# g.set_ylim(-10, 10)
plt.tight_layout()
plt.show()

if __name__ == '__main__':
    ...
