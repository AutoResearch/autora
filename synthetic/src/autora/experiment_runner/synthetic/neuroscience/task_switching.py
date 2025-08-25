from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def task_switching(
    name="Task Switching",
    resolution=50,
    priming_default=0.3,
    temperature=0.2,
    minimum_task_control=0.15,
    constant=1.5,
):
    """
    Task Switching

    Args:
        name: name of the experiment
        resolution: number of allowed values for stimulus
        priming_default: default for task priming
        temperature: temperature for softmax when computing performance of current task
        constant: constant for task activation
        minimum_task_control: minimum task control
    Examples:
        >>> s = task_switching()
        >>> s.run(np.array([[.5,.7,0]]), random_state=42)
           cur_task_strength  alt_task_strength  is_switch  cur_task_performance
        0                0.5                0.7        0.0              0.685351
    """

    params = dict(
        name=name,
        resolution=resolution,
        priming_default=priming_default,
        temperature=temperature,
        minimum_task_control=minimum_task_control,
        constant=constant,
    )

    current_task_strength = IV(
        name="cur_task_strength",
        allowed_values=np.linspace(1 / resolution, 1, resolution),  #
        value_range=(0, 1),
        units="intensity",
        variable_label="Strength of Current Task",
        type=ValueType.REAL,
    )

    alt_task_strength = IV(
        name="alt_task_strength",
        allowed_values=np.linspace(1 / resolution, 1, resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Strength of Alternative Task",
        type=ValueType.REAL,
    )

    is_switch = IV(
        name="is_switch",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="indicator",
        variable_label="Is Switch",
        type=ValueType.PROBABILITY_SAMPLE,
    )

    cur_task_performance = DV(
        name="cur_task_performance",
        value_range=(0, 1),
        units="performance",
        variable_label="Accuracy of Current Task",
        type=ValueType.PROBABILITY,
    )

    variables = VariableCollection(
        independent_variables=[current_task_strength, alt_task_strength, is_switch],
        dependent_variables=[cur_task_performance],
    )

    def inverse(x, A, B):
        y = 1 / (A * x + B)
        return y

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise: float = 0.01,
        random_state: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_state)
        X = np.array(conditions)
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            cur_task_strength = x[0]
            alt_task_strength = x[1]
            is_switch = x[2]

            # determine current task control

            input_ratio = (cur_task_strength + priming_default * (1 - is_switch)) / (
                alt_task_strength + priming_default * (is_switch)
            )

            cur_task_control = inverse(input_ratio, 2.61541389, 0.7042097)
            cur_task_control = np.max([cur_task_control, minimum_task_control])

            cur_task_input = (
                cur_task_strength
                + priming_default * (1 - is_switch)
                + cur_task_control
                + rng.normal(0, added_noise)
            )

            alt_task_input = (
                alt_task_strength
                + priming_default * (is_switch)
                + rng.normal(0, added_noise)
            )

            cur_task_activation = 1 - np.exp(-constant * cur_task_input)
            alt_task_activation = 1 - np.exp(-constant * alt_task_input)

            cur_task_performance = np.exp(cur_task_activation * 1 / temperature) / (
                np.exp(cur_task_activation * 1 / temperature)
                + np.exp(alt_task_activation * 1 / temperature)
            )

            Y[idx] = cur_task_performance
        experiment_data = pd.DataFrame(conditions)
        experiment_data.columns = [v.name for v in variables.independent_variables]
        experiment_data[variables.dependent_variables[0].name] = Y
        return experiment_data

    ground_truth = partial(run, added_noise=0.0)

    def domain():
        s1_values = variables.independent_variables[0].allowed_values
        s2_values = variables.independent_variables[1].allowed_values
        is_switch_values = variables.independent_variables[2].allowed_values
        X = np.array(np.meshgrid(s1_values, s2_values, is_switch_values)).T.reshape(-1, 3)
        # remove all combinations where s1 > s2
        # X = X[X[:, 0] <= X[:, 1]]
        return X

    def plotter(
        model=None,
    ):
        X = np.zeros((4, 3))

        # Values taken from Table 4 in Yeung & Monsell (2003)

        # word switch
        X[0, 0] = 0.5  # current task strength
        X[0, 1] = 0.1  # alternative task strength
        # X[0, 2] = 0.2 # current task control
        X[0, 2] = 1  # is switch

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

        y = ground_truth(X)

        word_switch_performance = y.at[0, 'cur_task_performance']
        word_repetition_performance = y.at[1, 'cur_task_performance']
        color_switch_performance = y.at[2, 'cur_task_performance']
        color_repetition_performance = y.at[3, 'cur_task_performance']

        x_data = [1, 2]
        word_performance = (
            1 - np.array([word_repetition_performance, word_switch_performance])
        ) * 100
        color_performance = (
            1 - np.array([color_repetition_performance, color_switch_performance])
        ) * 100

        if model is not None:
            y_pred = model.predict(X)
            word_switch_performance_pred = y_pred[0][0]
            word_repetition_performance_pred = y_pred[1][0]
            color_switch_performance_pred = y_pred[2][0]
            color_repetition_performance_pred = y_pred[3][0]
            word_performance_recovered = (
                1
                - np.array(
                    [word_repetition_performance_pred, word_switch_performance_pred]
                )
            ) * 100
            color_performance_recovered = (
                1
                - np.array(
                    [color_repetition_performance_pred, color_switch_performance_pred]
                )
            ) * 100

        legend = (
            "Word Task (Original)",
            "Color Task (Original)",
            "Word Task (Recovered)",
            "Color Task (Recovered)",
        )

        # plot
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())

        plt.plot(x_data, word_performance, label=legend[0], c=colors[col_keys[0]])
        plt.plot(x_data, color_performance, label=legend[1], c=colors[col_keys[1]])
        if model is not None:
            plt.plot(
                x_data,
                word_performance_recovered,
                "--",
                label=legend[2],
                c=colors[col_keys[0]],
            )
            plt.plot(
                x_data,
                color_performance_recovered,
                "--",
                label=legend[3],
                c=colors[col_keys[1]],
            )
        plt.xlim([0.5, 2.5])
        plt.ylim([0, 50])
        plt.ylabel("Error Rate (%)", fontsize="large")
        plt.legend(loc=2, fontsize="large")
        plt.title("Task Switching", fontsize="large")
        plt.xticks(x_data, ["Repetition", "Switch"], rotation="horizontal")
        plt.show()

    collection = SyntheticExperimentCollection(
        name=name,
        description=task_switching.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=task_switching,
    )
    return collection
