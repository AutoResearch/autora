import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# task switching parameters
task_switching_resolution = 100
priming_default = 0.3  # default for task priming
choice_temperature = 0.2 # temperature for softmax when computing performance of current task
minimum_task_control = 0.15 # minimum task control
c = 1.5  # constant for task activation
noise_sd = 0.01  # input noise standard deviation for task input # 0.1

# Task Switching Model by
# Yeung, N., & Monsell, S. (2003). Task switching and the control of response conflict. Psychological review, 110(4), 769.

def task_switching_metadata():

    current_task_strength = IV(
        name="cur_task_strength",
        allowed_values=np.linspace(0.05, 1, task_switching_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Strength of Current Task",
        type=ValueType.REAL
    )

    alt_task_strength = IV(
        name="alt_task_strength",
        allowed_values=np.linspace(0.05, 1, task_switching_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Strength of Alternative Task",
        type=ValueType.REAL
    )

    is_switch = IV(
        name="is_switch",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="indicator",
        variable_label="Is Switch",
        type=ValueType.PROBABILITY_SAMPLE
    )

    cur_task_performance = DV(
        name="cur_task_performance",
        value_range=(0, 1),
        units="performance",
        variable_label="Accuray of Current Task",
        type=ValueType.PROBABILITY
    )

    metadata = VariableCollection(
        independent_variables=[current_task_strength,
                               alt_task_strength,
                               is_switch],
        dependent_variables=[cur_task_performance],
    )

    return metadata

def task_switching_experiment(X: np.ndarray,
                             priming_constant: float = priming_default,
                             task_activation_constant: float = c,
                             choice_temperature: float =choice_temperature,
                             minimum_task_control: float = minimum_task_control,
                             std: float = noise_sd):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        cur_task_strength = x[0]
        alt_task_strength = x[1]
        is_switch = x[2]

        # determine current task control

        input_ratio = (cur_task_strength + priming_constant * (1-is_switch)) / \
                      (alt_task_strength + priming_constant * (is_switch))

        cur_task_control = inverse(input_ratio, 2.61541389, 0.7042097)
        cur_task_control = np.max([cur_task_control, minimum_task_control])

        cur_task_input = cur_task_strength + \
                         priming_constant * (1-is_switch) + \
                         cur_task_control + \
                         np.random.normal(0, std)

        alt_task_input = alt_task_strength + \
                         priming_constant * (is_switch) + \
                         np.random.normal(0, std)

        cur_task_activation = 1 - np.exp(-task_activation_constant * cur_task_input)
        alt_task_activation = 1 - np.exp(-task_activation_constant * alt_task_input)

        cur_task_performance = np.exp(cur_task_activation * 1/choice_temperature) / \
                               (np.exp(cur_task_activation * 1/choice_temperature) +
                                np.exp(alt_task_activation * 1/choice_temperature))

        # word switch
        # word nonswitch
        # color switch
        # color nonswitch

        Y[idx] = cur_task_performance


    return Y

def task_switching_data(metadata):

    cur_task_strength = metadata.independent_variables[0].allowed_values
    alt_task_strength = metadata.independent_variables[1].allowed_values
    is_switch = metadata.independent_variables[2].allowed_values

    X = np.array(np.meshgrid(cur_task_strength,
                             alt_task_strength,
                             is_switch)).T.reshape(-1,3)

    y = task_switching_experiment(X, std=0)

    return X, y


def plot_task_switching(model = None):
    import matplotlib.pyplot as plt

    X = np.zeros((4, 3))

    # Values taken from Table 4 in Yeung & Monsell (2003)

    # word switch
    X[0, 0] = 0.5 # current task strength
    X[0, 1] = 0.1 # alternative task strength
    # X[0, 2] = 0.2 # current task control
    X[0, 2] = 1 # is switch

    # word repetition
    X[1, 0] = 0.5  # current task strength
    X[1, 1] = 0.1  # alternative task strength
    # X[1, 2] = 0.15 # current task control
    X[1, 2] = 0  # is switch

    # color switch
    X[2, 0] = 0.1  # current task strength
    X[2, 1] = 0.5  # alternative task strength
    # X[2, 2] = 0.97  # current task control
    X[2, 2] = 1  # is switch

    # color repetition
    X[3, 0] = 0.1  # current task strength
    X[3, 1] = 0.5  # alternative task strength
    # X[3, 2] = 0.38  # current task control
    X[3, 2] = 0  # is switch

    y = task_switching_experiment(X, priming_constant=0.3, std=0)

    word_switch_performance = y[0,0]
    word_repetition_performance = y[1,0]
    color_switch_performance = y[2,0]
    color_repetition_performance = y[3,0]

    x_data = [1, 2]
    word_performance = (1 - np.array([word_repetition_performance,
                                      word_switch_performance])) * 100
    color_performance = (1 - np.array([color_repetition_performance,
                                       color_switch_performance])) * 100

    if model is not None:
        y_pred = model.predict(X)
        word_switch_performance_pred = y_pred[0]
        word_repetition_performance_pred = y_pred[1]
        color_switch_performance_pred = y_pred[2]
        color_repetition_performance_pred = y_pred[3]
        word_performance_recovered = (1 - [word_repetition_performance_pred,
                                      word_switch_performance_pred]) * 100
        color_performance_recovered = (1 - [color_repetition_performance_pred,
                                       color_switch_performance_pred]) * 100

    legend = ('Word Task (Original)', 'Color Task (Original)',
              'Word Task (Recovered)', 'Color Task (Recovered)',)

    # plot
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = mcolors.TABLEAU_COLORS
    col_keys = list(colors.keys())


    plt.plot(x_data, word_performance, label=legend[0], c=colors[col_keys[0]])
    plt.plot(x_data, color_performance, label=legend[1], c=colors[col_keys[1]])
    if model is not None:
        plt.plot(x_data, word_performance_recovered, '--', label=legend[2], c=colors[col_keys[0]])
        plt.plot(x_data, color_performance_recovered, '--', label=legend[3], c=colors[col_keys[1]])
    plt.xlim([0.5 , 2.5])
    plt.ylim([0, 50])
    plt.ylabel("Error Rate (%)", fontsize="large")
    plt.legend(loc=2, fontsize="large")
    plt.title("Task Switching", fontsize="large")
    plt.xticks(x_data, ['Repetition', 'Switch'], rotation='horizontal')
    plt.show()

def inverse(x, A, B):
    y = 1/(A*x+B)
    return y

def fit_control_strength():
    from scipy.optimize import curve_fit

    x = [8, 1.25, 0.8, 0.125]
    y = [0.15, 0.2, 0.38, 0.97]

    parameters, covariance = curve_fit(inverse, x, y)
    print(parameters)

    x_pred = np.linspace(0, 10, 100)
    y_pred = inverse(x_pred, *parameters)

    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o')
    plt.plot(x_pred, y_pred)
    plt.show()

# X, y = task_switching_data(task_switching_metadata())
# plot_task_switching()




