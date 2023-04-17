import pandas as pd
from autora.skl.bms import BMSRegressor
from sklearn.metrics import log_loss
from scipy.special import expit
import numpy as np
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
    # if y_target.shape[1] == 1:
    #     y_target = y_target.flatten()
    if y_prediction.shape[1] == 1:
        y_prediction = y_prediction.flatten()

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

bms_theorists = [
    'Root Fixed',
    'Root Variable',
    'Regular Fixed',
    'Regular Variable',
]


def run_fleming(iter: int = 1, bms_theorist: str = 'Regular Fixed'):
    np.seed(iter)
    # read in dataset for main experiment
    data = pd.read_csv('../baseline_datasets/data_fleming/mainexp-switch-effort.csv',
                       low_memory=False)
    data = filter_data(data)
    data = rename_data(data)
    grouping_variable = 'g'

    test_size = 0.2

    # separate into training and validation
    data_train = pd.DataFrame(columns=data.columns)
    data_test = pd.DataFrame(columns=data.columns)
    for participant in data[grouping_variable].unique():
        data_par = data.loc[data[grouping_variable] == participant]
        test_num = int(data_par.shape[0] * test_size)
        data_test = data_test.append(data_par.head(test_num))
        data_train = data_train.append(data_par.head(data_par.shape[0] - test_num))
        print(participant)

    # Initialize models
    # bms = BMSRegressor(epochs=1500)
    # log_mod = FlemingTheorist()

    # DataFrame to hold final results
    results = pd.DataFrame(columns=['Theorist', 'Loss'])

    # Fit and Test Hierarchical Logistic Regression models
    # for cond in logistic_models.keys():
    #     log_mod = FlemingTheorist()
    #     fitted_log_model = log_mod.fit(data=data, X_cond=logistic_models[cond], y_col='y')
    #     y_predict = fitted_log_model.predict(data)
    #     loss = log_loss(y_true=data['y'], y_pred=y_predict)
    #     results.append(pd.Series(['Logistic Regression '+str(cond), loss]), ignore_index=True)

    # Fit and Test BMS models
    # for bms_theorist in bms_theorists:
    root = None
    root_string = ''
    if 'Root' in bms_theorist:
        root = sigmoid
        root_string = 'Rooted'
    if 'Fixed' in bms_theorist:
        bms = BMSRegressor(epochs=1500)
        bms.fit(X=data_train[['E', 'E2', 'R', 'R2']], y=data_train['y'], root=root)
        y_predict = bms.predict(data_test[['E', 'E2', 'R', 'R2']])
        loss = log_loss(y_true=data_test['y'], y_pred=y_predict)
        results.append(pd.Series(['BMS Fixed' + root_string, loss]), ignore_index=True)
    else:
        bms = BMSRegressor(epochs=400)  # less epochs for much smaller data
        sle = 0
        num_par = 0
        for participant in data[grouping_variable].unique():
            df_ind = data.loc[data[grouping_variable] == participant]
            test_num = int(df_ind.shape[0] * test_size)
            data_test = df_ind.head(test_num)
            data_train = df_ind.head(df_ind.shape[0] - test_num)
            num_trials = df_ind.shape[0]
            if len(data_test['y'].unique()) == 1:
                print(
                    'rejecting participant ' + str(participant) + 'from training due to either'
                                                                  'always or never accepting')
            else:
                bms.fit(X=data_test[['E', 'E2', 'R', 'R2']], y=data_test['y'], root=root)
                print(bms.model_)
                y_predict = bms.predict(data_test[['E', 'E2', 'R', 'R2']])
                loss = log_loss(y_true=data_test['y'], y_pred=y_predict)
                num_par += test_num
                # sse = get_MSE(theorist=bms, y_target=data_test['y'],
                #               x=data_test[['E', 'E2', 'R', 'R2']]) * test_num
                sle += loss * test_num
        mle = sle / num_par
        results.append(pd.Series(['BMS Variable' + root_string, mle]), ignore_index=True)

    # Save study data to csv
    results.to_csv('baseline_models/fleming_implementation_'+str(iter)+'.csv', index=False)


if __name__ == '__main__':
    run_fleming()
