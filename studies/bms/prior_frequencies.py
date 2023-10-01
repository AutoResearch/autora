import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def __get_frequencies(prior_name):
    freq_dict = {
        'Guimera2020': {
            'sinh': float(5/4080),
            'cos': float(65/4080),
            'log': float(132/4080),
            'tanh': float(6/4080),
            'pow2': float(547/4080),
            '-': float(520/4080),
            'abs': float(27/4080),
            'sqrt': float(130/4080),
            'cosh': float(4/4080),
            'fac': float(7/4080),
            '+': float(1271/4080),
            '**': float(652/4080),
            'exp': float(129/4080),
            'pow3': float(38/4080),
            '*': float(2774/4080),
            '/': float(1146/4080),
            'sin': float(39/4080),
            'tan': float(4/4080)
        },
        'Default': {
            '+': float(1271 / 4080),
            '-': float(520 / 4080),
            '*': float(2774 / 4080),
            '/': float(1146 / 4080),
            '**': float(652 / 4080),
            'log': float(132 / 4080),
            'exp': float(129 / 4080),
            'sin': float(39 / 4080),
            'cos': float(65 / 4080),
        },
        'Uniform': {
            '+': float(750 / 4080),
            '-': float(750 / 4080),
            '*': float(750 / 4080),
            '/': float(750 / 4080),
            '**': float(750 / 4080),
            'log': float(750 / 4080),
            'exp': float(750 / 4080),
            'sin': float(750 / 4080),
            'cos': float(750 / 4080),
        },
        'Williams2023Psychophysics': {
            "+": float(29 / 48),
            "-": float(35 / 48),
            "*": float(66 / 48),
            "/": float(14 / 48),
            "**": float(10 / 48),
            "log": float(4 / 48),
            "exp": float(0.1 / 48),
            "sin": float(0.1 / 48),
            "cos": float(0.1 / 48),
        },
        'Williams2023CognitivePsychology': {
            "+": float(65 / 92),
            "-": float(66 / 92),
            "*": float(120 / 92),
            "/": float(37 / 92),
            "**": float(33 / 92),
            "log": float(8 / 92),
            "exp": float(3 / 92),
            "sin": float(0.1 / 92),
            "cos": float(1 / 92),
        },
        'Williams2023BehavioralEconomics': {
            "+": float(46 / 115),
            "-": float(54 / 115),
            "*": float(105 / 115),
            "/": float(46 / 115),
            "**": float(38 / 115),
            "log": float(8 / 115),
            "exp": float(3 / 115),
            "sin": float(0.1 / 115),
            "cos": float(0.1 / 115),
        },
        'Williams2023SUPERCognitiveScienceNew': {
            "*": 0.7569786535303776,
            "+": 0.10344827586206896,
            "**": 0.14942528735632185,
            "/": 0.03776683087027915,
            "-": 0.11986863711001643,
            "pow2": 0.20361247947454844,
            "abs": 0.021346469622331693,
            "sqrt": 0.014778325123152709,
            "pow3": 0.041050903119868636,
            "log": 0.019704433497536946,
            "max": 0.0049261083743842365
        },
        'Williams2023SUPERCognitivePsychologyNew': {
            "*": 0.8266666666666667,
            "-": 0.2,
            "**": 0.05333333333333334,
            "+": 0.26666666666666666,
            "/": 0.06666666666666667,
            "pow2": 0.29333333333333333,
            "sqrt": 0.06666666666666667,
            "max": 0.02666666666666667,
            "log": 0.05333333333333334,
            "pow3": 0.013333333333333334
        },
        'Williams2023SUPERMaterialsScienceNew': {
            "*": 1.3944264690533348,
            "+": 0.1584385763490241,
            "**": 0.18515812545663293,
            "/": 0.13745955537000312,
            "-": 0.2161569773510072,
            "pow2": 0.3659325748877988,
            "abs": 0.01962216887590022,
            "cos": 0.023692725185262498,
            "sqrt": 0.06721636572382841,
            "atan": 0.0015655985805239536,
            "pow3": 0.07640121072956894,
            "exp": 0.009602337960546915,
            "sin": 0.01586473228264273,
            "tan": 0.003548690115854295,
            "log": 0.002504957728838326,
            "sinh": 0.00031311971610479075,
            "cosh": 0.0005218661935079845,
            "max": 0.00010437323870159691,
            "tanh": 0.0007306126709111783,
            "asin": 0.00020874647740319382,
            "min": 0.00031311971610479075,
            "acos": 0.00031311971610479075
        },
        'Williams2023SUPERNeuroscienceNew': {
            "*": 0.8642745709828393,
            "-": 0.21528861154446177,
            "+": 0.1794071762870515,
            "**": 0.1731669266770671,
            "/": 0.1029641185647426,
            "pow2": 0.1606864274570983,
            "abs": 0.0062402496099844,
            "sqrt": 0.0405616224648986,
            "max": 0.0031201248049922,
            "log": 0.0109204368174727,
            "pow3": 0.078003120124805,
            "exp": 0.0187207488299532,
            "cos": 0.0031201248049922,
            "tanh": 0.0046801872074883,
            "sin": 0.0015600624024961
        },
        'Williams2023SUPERAverageNew': {
            'log': 0.00019685698967966762,
            '**': 0.008293124246079614,
            'sinh': 1.2565339766787294e-05,
            '+': 0.007187374346602332,
            'tan': 0.00014240718402358933,
            'cosh': 2.094223294464549e-05,
            'sqrt': 0.002864897466827503,
            'asin': 8.376893177858196e-06,
            'tanh': 4.188446588929098e-05,
            '*': 0.06046860340436939,
            '/': 0.005909898136978957,
            '-': 0.009620861814770138,
            'abs': 0.0008586315507304651,
            'pow2': 0.015727616941428765,
            'cos': 0.0009591542688647634,
            'max': 3.3507572711432785e-05,
            'atan': 6.282669883393646e-05,
            'exp': 0.0004355984452486262,
            'acos': 1.2565339766787294e-05,
            'pow3': 0.003384264843854711,
            'min': 1.2565339766787294e-05,
            'sin': 0.000640832328106152
        },
        'Williams2023SUPERUniformNew': {
            'max': 0.005155067219104561,
            '-': 0.005155067219104561,
            '/': 0.005155067219104561,
            'cos': 0.005155067219104561,
            'tan': 0.005155067219104561,
            'cosh': 0.005155067219104561,
            '*': 0.005155067219104561,
            'log': 0.005155067219104561,
            'pow3': 0.005155067219104561,
            'acos': 0.005155067219104561,
            'min': 0.005155067219104561,
            'abs': 0.005155067219104561,
            'tanh': 0.005155067219104561,
            'pow2': 0.005155067219104561,
            'sqrt': 0.005155067219104561,
            'sinh': 0.005155067219104561,
            'asin': 0.005155067219104561,
            'atan': 0.005155067219104561,
            'exp': 0.005155067219104561,
            '+': 0.005155067219104561,
            '**': 0.005155067219104561,
            'sin': 0.005155067219104561,
        }
    }
    if prior_name in freq_dict.keys():
        return freq_dict[prior_name]
    else:
        raise KeyError('This prior is not available')


def __get_ops():
    ops = {
        "sin": 1,
        "cos": 1,
        "tan": 1,
        "exp": 1,
        "log": 1,
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        "pow2": 1,
        "pow3": 1,
        "abs": 1,
        "sqrt": 1,
        "fac": 1,
        "-": 1,
        "+": 2,
        "*": 2,
        "/": 2,
        "**": 2,
        "sig": 1,
        "max": 1,
        "acos": 1,
        "asin": 1,
        "atan": 1,
        "min": 1
    }
    return ops


def get_raw_priors(prior="Guimera2020"):
    priors = __get_frequencies(prior)
    all_ops = __get_ops()
    ops = {k: v for k, v in all_ops.items() if k in priors}
    return priors, ops


def plot_frequencies(prior_names):
    x = pd.DataFrame(columns=['Prior', 'Operation', 'Frequency'])
    for name in prior_names:
        dic = get_raw_priors(name)[0]
        df_dic = pd.DataFrame.from_dict(dic, orient='index', columns=['Frequency'])
        df_dic['Prior'] = name
        df_dic.reset_index(inplace=True)
        df_dic = df_dic.rename(columns={'index': 'Operation'})
        x = pd.concat([x, df_dic], ignore_index=True)
    df_dic.reset_index(inplace=True)
    print(x)
    x['Log-Frequency'] = x['Frequency'].apply(np.log)
    print(x)
    sns.lineplot(data=x, x='Operation', y='Frequency', hue='Prior')
    plt.show()
    sns.lineplot(data=x, x='Operation', y='Log-Frequency', hue='Prior')
    plt.show()


if __name__ == '__main__':
    prior_names = ['Default',
                   'Williams2023Psychophysics',
                   'Williams2023CognitivePsychology',
                   'Williams2023BehavioralEconomics',
                   'Uniform',]
    plot_frequencies(prior_names)
