from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def weber_fechner_law(
    name="Weber-Fechner Law",
    resolution=100,
    constant=1.0,
    maximum_stimulus_intensity=5.0,
):
    """
    Weber-Fechner Law

    Args:
        name: name of the experiment
        resolution: number of allowed values for stimulus 1 and 2
        constant: constant multiplier
        maximum_stimulus_intensity: maximum value for stimulus 1 and 2

    Examples:
        >>> experiment = weber_fechner_law()

        # The runner can accept numpy arrays or pandas DataFrames, but the return value will
        # always be a pandas DataFrame.
        >>> experiment.run(np.array([[.1,.2]]), random_state=42)
            S1   S2  difference_detected
        0  0.1  0.2             0.696194

        >>> experiment.run(pd.DataFrame({'S1': [0.1], 'S2': [0.2]}), random_state=42)
            S1   S2  difference_detected
        0  0.1  0.2             0.696194

    """

    params = dict(
        name=name,
        resolution=resolution,
        constant=constant,
        maximum_stimulus_intensity=maximum_stimulus_intensity,
    )

    iv1 = IV(
        name="S1",
        allowed_values=np.linspace(
            1 / resolution, maximum_stimulus_intensity, resolution
        ),
        value_range=(1 / resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus 1 Intensity",
        type=ValueType.REAL,
    )

    iv2 = IV(
        name="S2",
        allowed_values=np.linspace(
            1 / resolution, maximum_stimulus_intensity, resolution
        ),
        value_range=(1 / resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus 2 Intensity",
        type=ValueType.REAL,
    )

    dv1 = DV(
        name="difference_detected",
        value_range=(0, maximum_stimulus_intensity),
        units="sensation",
        variable_label="Sensation",
        type=ValueType.REAL,
    )

    variables = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise=0.01,
        random_state: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_state)
        X = np.array(conditions)

        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            y = constant * np.log(x[1] / x[0]) + rng.normal(0, added_noise)
            Y[idx] = y

        experiment_data = pd.DataFrame(conditions)
        experiment_data.columns = [v.name for v in variables.independent_variables]
        experiment_data[dv1.name] = Y
        return experiment_data

    ground_truth = partial(run, added_noise=0.0)

    def domain():
        s1_values = variables.independent_variables[0].allowed_values
        s2_values = variables.independent_variables[1].allowed_values
        X = np.array(np.meshgrid(s1_values, s2_values)).T.reshape(-1, 2)
        # remove all combinations where s1 > s2
        X = X[X[:, 0] <= X[:, 1]]
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())

        S0_list = [1, 2, 4]
        delta_S = np.linspace(0, 5, 100)

        for idx, S0_value in enumerate(S0_list):
            S0 = S0_value + np.zeros(delta_S.shape)
            S1 = S0 + delta_S
            X = np.array([S0, S1]).T
            y = ground_truth(X)[dv1.name]
            plt.plot(
                delta_S,
                y,
                label=f"$S_0 = {S0_value}$ (Original)",
                c=colors[col_keys[idx]],
            )
            if model is not None:
                y = model.predict(X)
                plt.plot(
                    delta_S,
                    y,
                    label=f"$S_0 = {S0_value}$ (Recovered)",
                    c=colors[col_keys[idx]],
                    linestyle="--",
                )

        x_limit = [0, variables.independent_variables[0].value_range[1]]
        y_limit = [0, 2]
        x_label = r"Stimulus Intensity Difference $\Delta S = S_1 - S_0$"
        y_label = "Perceived Intensity of Stimulus $S_1$"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=2, fontsize="medium")
        plt.title("Weber-Fechner Law", fontsize="x-large")

    collection = SyntheticExperimentCollection(
        name=name,
        description=weber_fechner_law.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=weber_fechner_law,
    )
    return collection
