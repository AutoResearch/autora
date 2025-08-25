"""
A template synthetic experiment.

Examples:
    >>> from autora.experiment_runner.synthetic.abstract.template_experiment import (
    ...     template_experiment
    ... )

    We can instantiate the experiment using the imported function
    >>> s = template_experiment()
    >>> s  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    SyntheticExperimentCollection(name='Template Experiment', description='...',
        params={'name': ...}, ...)

    >>> s.name
    'Template Experiment'

    >>> s.variables # doctest: +ELLIPSIS
    VariableCollection(...)

    >>> s.domain()
    array([[0],
           [1],
           [2],
           [3]])

    >>> s.ground_truth  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    functools.partial(<function template_experiment.<locals>.run at 0x...>,
                      added_noise=0.0)

    >>> float(s.ground_truth(1.))
    2.0

    >>> s.ground_truth(s.domain())
    array([[1.],
           [2.],
           [3.],
           [4.]])


    >>> s.run  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <function template_experiment.<locals>.run at 0x...>

    >>> float(s.run(1., random_state=42))
    2.003047170797544

    >>> s.run(s.domain(), random_state=42)
    array([[1.00304717],
           [1.98960016],
           [3.00750451],
           [4.00940565]])

    >>> s.plotter()
    >>> plt.show()  # doctest: +SKIP

    Generate a new version of the experiment with different parameters:
    >>> new_params = dict(s.params)
    >>> s.factory_function(**new_params)  # doctest: +ELLIPSIS
    SyntheticExperimentCollection(...)

"""


from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, VariableCollection


def template_experiment(
    # Add any configurable parameters with their defaults here:
    name: str = "Template Experiment",
):
    """
    A template for synthetic experiments.

    Parameters:
        name: name of the experiment
    """

    params = dict(
        # Include all parameters here:
        name=name,
    )

    # Define variables
    x = IV(name="Intensity", allowed_values=np.arange(4))
    y = DV(name="Response")
    variables = VariableCollection(
        independent_variables=[x],
        dependent_variables=[y],
    )

    # Define experiment runner

    def run(
        conditions: ArrayLike,
        added_noise: float = 0.01,
        random_state: Optional[int] = None,
    ):
        """A function which simulates noisy observations."""
        rng = np.random.default_rng(random_state)
        x_ = np.array(conditions)
        y = x_ + 1.0 + rng.normal(0, added_noise, size=x_.shape)
        return y

    ground_truth = partial(run, added_noise=0.0)
    """A function which simulates perfect observations"""

    def domain():
        """A function which returns all possible independent variable values as a 2D array."""
        x = variables.independent_variables[0].allowed_values.reshape(-1, 1)
        return x

    def plotter(model=None):
        """A function which plots the ground truth and (optionally) a fitted model."""
        import matplotlib.pyplot as plt

        plt.figure()
        x = domain()
        plt.plot(x, ground_truth(x), label="Ground Truth")

        if model is not None:
            plt.plot(x, model.predict(x), label="Fitted Model")

        plt.xlabel(variables.independent_variables[0].name)
        plt.ylabel(variables.dependent_variables[0].name)
        plt.legend()
        plt.title(name)

    # The object which gets stored in the synthetic inventory
    collection = SyntheticExperimentCollection(
        name=name,
        description=template_experiment.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=template_experiment,
    )
    return collection
