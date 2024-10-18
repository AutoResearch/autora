from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def exp_learning(
    name="Exponential Learning",
    resolution=100,
    minimum_trial=1,
    minimum_initial_value=0,
    maximum_initial_value=0.5,
    lr=0.03,
    p_asymptotic=1.0,
):
    """
    Exponential Learning

    Args:
        p_asymptotic: additive bias on constant multiplier
        lr: learning rate
        maximum_initial_value: upper bound for initial p value
        minimum_initial_value: lower bound for initial p value
        minimum_trial: upper bound for exponential constant
        name: name of the experiment
        resolution: number of allowed values for stimulus
        Examples:
        >>> s = exp_learning()
        >>> s.run(np.array([[.2,.1]]), random_state=42)
           P_asymptotic  trial  performance
        0           0.2    0.1     0.205444
    """

    maximum_trial = resolution

    params = dict(
        name="Exponential Learning",
        resolution=resolution,
        minimum_trial=minimum_trial,
        maximum_trial=maximum_trial,
        minimum_initial_value=minimum_initial_value,
        maximum_initial_value=maximum_initial_value,
        lr=lr,
        p_asymptotic=p_asymptotic,
    )

    p_initial = IV(
        name="P_asymptotic",
        allowed_values=np.linspace(
            minimum_initial_value, maximum_initial_value, resolution
        ),
        value_range=(minimum_initial_value, maximum_initial_value),
        units="performance",
        variable_label="Asymptotic Performance",
        type=ValueType.REAL,
    )

    trial = IV(
        name="trial",
        allowed_values=np.linspace(minimum_trial, maximum_trial, resolution),
        value_range=(minimum_trial, maximum_trial),
        units="trials",
        variable_label="Trials",
        type=ValueType.REAL,
    )

    performance = DV(
        name="performance",
        value_range=(0, p_asymptotic),
        units="performance",
        variable_label="Performance",
        type=ValueType.REAL,
    )

    variables = VariableCollection(
        independent_variables=[p_initial, trial],
        dependent_variables=[performance],
    )

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise: float = 0.01,
        random_state: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_state)
        X = np.array(conditions)
        Y = np.zeros((X.shape[0], 1))

        # exp learning function according to
        # Heathcote, A., Brown, S., & Mewhort, D. J. (2000). The power law repealed:
        # The case for an exponential law of practice. Psychonomic bulletin & review, 7(2), 185–207.

        # Thurstone, L. L. (1919). The learning curve equation.
        # Psy- chological Monographs, 26(3), i.

        for idx, x in enumerate(X):
            p_initial_exp = x[0]
            trial_exp = x[1]
            y = (
                p_asymptotic
                - (p_asymptotic - p_initial_exp) * np.exp(-lr * trial_exp)
                + rng.normal(0, added_noise)
            )
            Y[idx] = y

        experiment_data = pd.DataFrame(conditions)
        experiment_data.columns = [v.name for v in variables.independent_variables]
        experiment_data[variables.dependent_variables[0].name] = Y
        return experiment_data

    ground_truth = partial(run, added_noise=0.0)

    def domain():
        p_initial_values = variables.independent_variables[0].allowed_values
        trial_values = variables.independent_variables[1].allowed_values

        X = np.array(np.meshgrid(p_initial_values, trial_values)).T.reshape(-1, 2)
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.pyplot as plt

        P_0_list = [0, 0.25, 0.5]

        for P_0 in P_0_list:
            X = np.zeros((len(trial.allowed_values), 2))
            X[:, 0] = P_0
            X[:, 1] = trial.allowed_values

            dvs = [dv.name for dv in variables.dependent_variables]
            y = ground_truth(X)[dvs]

            plt.plot(trial.allowed_values, y, label=f"$P_0 = {P_0}$ (Original)")
            if model is not None:
                y = model.predict(X)
                plt.plot(trial.allowed_values, y, label=f"$P_0 = {P_0}$ (Recovered)", linestyle="--")

        x_limit = [0, variables.independent_variables[1].value_range[1]]
        y_limit = [0, 1]
        x_label = "Trial $t$"
        y_label = "Performance $P_n$"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=4, fontsize="medium")
        plt.title("Exponential Learning", fontsize="x-large")
        plt.show()

    collection = SyntheticExperimentCollection(
        name=name,
        description=exp_learning.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=exp_learning,
    )
    return collection
