import pandas as pd
from autora.skl.bms import BMSRegressor
from scipy.special import expit
import numpy as np
from random import seed
from studies.cogsci2023.fleming_theorist import FlemingTheorist


def sigmoid(x):
    return expit(x)


def filter_data(df):
    # filter out rows with Nans
    df = df.loc[df['accept'].notna()]

    # filter out irrelevant columns
    df = df[['Participant Private ID', 'branch-ar27', 'offer_points', 'offer_difficulty', 'accept']]

    # Remove participants that were excluded from the study
    df = df.loc[df['branch-ar27'] == 'Completed full task']
    df = df.loc[~df['Participant Private ID'].isin([81, 215, 223, 34, 147, 58, 286])]
    assert len(
        df['Participant Private ID'].unique()) == 290  # final reported number of participants
    return df


def rename_data(data):
    data['y'] = data['accept']
    data['g'] = data['Participant Private ID']
    data['E'] = data['offer_difficulty']
    data['R'] = data['offer_points']
    data['E2'] = np.square(data['E'])
    data['R2'] = np.square(data['R'])
    data = data.drop(labels=['Participant Private ID', 'branch-ar27', 'offer_points', 'offer_difficulty', 'accept'], axis=1)
    print(data.columns)
    return data


def get_MSE(theorist, x, y_target):
    y_prediction = theorist.predict(x)
    if y_prediction.shape[1] == 1:
        y_prediction = y_prediction.flatten()
    assert y_target.shape == y_prediction.shape
    MSE = np.mean(np.square(y_target - y_prediction))

    return MSE


logistic_models = {
    1: {'1': 'f', 'E': None, 'E2': None, 'R': 'f', 'R2': None},
    2: {'1': 'f', 'E': 'f', 'E2': None, 'R': 'f', 'R2': None},
    3: {'1': 'v', 'E': 'f', 'E2': None, 'R': 'v', 'R2': None},
    4: {'1': 'v', 'E': 'v', 'E2': None, 'R': 'f', 'R2': None},
    5: {'1': 'f', 'E': 'f', 'E2': 'f', 'R': 'f', 'R2': None},
    6: {'1': 'v', 'E': 'f', 'E2': 'f', 'R': 'v', 'R2': None},
    7: {'1': 'v', 'E': 'v', 'E2': 'v', 'R': None, 'R2': None},
    8: {'1': 'v', 'E': None, 'E2': 'v', 'R': 'v', 'R2': None},
    9: {'1': 'v', 'E': 'v', 'E2': None, 'R': 'v', 'R2': 'v'},
    10: {'1': 'v', 'E': 'v', 'E2': None, 'R': None, 'R2': 'v'},
    11: {'1': 'v', 'E': 'v', 'E2': 'v', 'R': 'v', 'R2': 'v'},
    12: {'1': 'v', 'E': None, 'E2': 'v', 'R': None, 'R2': 'v'}
    }

# theorists = [
#     'BMS Root Fixed',
#     'BMS Root Variable',
#     'BMS Regular Fixed',
#     'BMS Regular Variable',
# ]


def run_fleming(iter: int = 1, theorist: str = 'BMS Regular Fixed'):
    seed(iter)
    # read in dataset for main experiment
    data = pd.read_csv('../baseline_datasets/data_fleming/mainexp-switch-effort.csv',
                       low_memory=False)
    data = filter_data(data)
    data = rename_data(data)
    if "BMS" in theorist and "Class" not in theorist:
        data = data.groupby(["g", "R", "E"]).mean().reset_index()
    grouping_variable = "g"

    # test_size = 0.2
    # separate into training and validation
    # data_train = pd.DataFrame(columns=data.columns)
    # data_test = pd.DataFrame(columns=data.columns)
    # for participant in data[grouping_variable].unique():
    #     data_par = data.loc[data[grouping_variable] == participant]
    #     test_num = int(data_par.shape[0] * test_size)
    #     data_test = data_test.append(data_par.head(test_num))
    #     data_train = data_train.append(data_par.head(data_par.shape[0] - test_num))
    #     print(participant)
    data_train = data
    data_test = data

    # DataFrame to hold final results
    predictions = pd.DataFrame(columns=['Theorist', 'Participant', 'Ground Truth', 'Predictions'])

    # Fit and Test BMS models
    if 'BMS' in theorist:
        root = None
        root_string = ''
        if 'Root' in theorist:
            root = sigmoid
            root_string = 'Rooted'
        bms = BMSRegressor(epochs=30)
        if 'Class' in theorist:
            bms.fit(X=data_train[['E', 'E2', 'R', 'R2']], y=data_train['y'], root=root, data_type='class')
        else:
            bms.fit(X=data_train[['E', 'E2', 'R', 'R2']], y=data_train['y'], root=root)
        y_predict = bms.predict(data_test[['E', 'E2', 'R', 'R2']])
        predictions.append(pd.DataFrame(data=[['BMS Fixed' + root_string for _ in y_predict],
                                              [0 for _ in y_predict],
                                              y_predict]), ignore_index=True)
    else:
        # Fit and Test Hierarchical Logistic Regression models
        cond = int(theorist)
        log_mod = FlemingTheorist()
        fitted_log_model = log_mod.fit(data=data, X_cond=logistic_models[cond], y_col='y')
        y_predict = fitted_log_model.predict(data)

    predictions['Ground Truth'] = data_test['y']
    predictions['Predictions'] = y_predict
    predictions['Participant'] = data[grouping_variable]
    predictions['Theorist'] = theorist

    # Save study data to csv
    predictions.to_csv('../baseline_models/fleming_implementation_'+theorist+'_'+str(iter)+'.csv', index=False)


if __name__ == '__main__':
    run_fleming(theorist='1')

# if len(data_test['y'].unique()) == 1:
            #    print(
            #        'rejecting participant ' + str(participant) + 'from training due to either'
            #                                                      'always or never accepting')
            # else:

# else:
#     bms = BMSRegressor(epochs=10)  # less epochs for much smaller data
#     num_par = 0
#     for participant in data[grouping_variable].unique():
#         df_ind = data.loc[data[grouping_variable] == participant]
#         test_num = int(df_ind.shape[0] * test_size)
#         data_test = df_ind.head(test_num)
#         bms.fit(X=data_test[['E', 'E2', 'R', 'R2']], y=data_test['y'], root=root)
#         print(bms.model_)
#         y_predict = bms.predict(data_test[['E', 'E2', 'R', 'R2']])
#         predictions.append(pd.DataFrame(data=[['BMS Variable' + root_string for _ in y_predict],
#                                               [participant for _ in y_predict],
#                                               y_predict]), ignore_index=True)
#         num_par += test_num
