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
        "relu": 1,
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
