"""
Module for registering and retrieving synthetic models from an inventory.

Examples:
    To add and recover a new model from the inventory, we need to define it using a function
    (closure).
    We start by importing the modules we'll need:
    >>> from functools import partial
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from autora.synthetic.inventory import SyntheticExperimentCollection
    >>> from autora.synthetic import register, retrieve
    >>> from autora.variable import IV, DV, VariableCollection

    Then we can define the function. We define all the arguments we need for the
    SyntheticExperimentCollection, and then instantiate that collection. The closure – in this
    case `sinusoid_experiment` – is the scope for all the parameters we need.
    >>> def sinusoid_experiment(omega=np.pi/3, delta=np.pi/2., m=0.3, resolution=1000,
    ...                         rng=np.random.default_rng()):
    ...     name = "Shifted Sinusoid"
    ...
    ...     params = dict(omega=omega, delta=delta, resolution=resolution, m=m, rng=rng)
    ...
    ...     x = IV(name="x", value_range=(-6 * np.pi, 6 * np.pi))
    ...     y = DV(name="y", value_range=(-1, 1))
    ...     metadata = VariableCollection(independent_variables=[x], dependent_variables=[y])
    ...
    ...     def domain():
    ...         return np.linspace(*x.value_range, resolution).reshape(-1, 1)
    ...
    ...     def experiment_runner(X, std=0.1):
    ...         return np.sin((X - delta) * omega) + (X * m) + rng.normal(0, std, X.shape)
    ...
    ...     def ground_truth(X):
    ...         return experiment_runner(X, std=0.)
    ...
    ...     def plotter(model=None):
    ...         plt.plot(domain(), ground_truth(domain()), label="Ground Truth")
    ...         if model is not None:
    ...             plt.plot(domain(), model.predict(domain()), label="Model")
    ...         plt.title(name)
    ...
    ...     collection = SyntheticExperimentCollection(
    ...         name=name,
    ...         params=params,
    ...         metadata=metadata,
    ...         domain=domain,
    ...         experiment_runner=experiment_runner,
    ...         ground_truth=ground_truth,
    ...         plotter=plotter,
    ...     )
    ...
    ...     return collection

    Then we can register the experiment. We register the function, rather than evaluating it.
    >>> register("sinusoid_experiment", sinusoid_experiment)

    When we want to retrieve the experiment, we can just use the default values if we like:
    >>> s = retrieve("sinusoid_experiment")

    The object returned includes all the used parameters as a dictionary
    >>> s.params  # doctest: +ELLIPSIS
    {'omega': 1.0..., 'delta': 1.5..., 'resolution': 1000, 'm': 0.3, ...}

    If we need to modify the parameter values, we can pass them as arguments to the retrieve
    function:
    >>> t = retrieve("sinusoid_experiment", delta=0.2)
    >>> t.params  # doctest: +ELLIPSIS
    {..., 'delta': 0.2, ...}
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from autora.variable import VariableCollection


class SyntheticExperimentClosure(Protocol):
    """A function which returns a SyntheticExperimentCollection."""

    def __call__(self, *args, **kwargs) -> SyntheticExperimentCollection:
        ...


class SupportsPredict(Protocol):
    def predict(self, X) -> Any:
        ...


@dataclass
class SyntheticExperimentCollection:
    """
    Represents a synthetic experiment.

    Attributes:
        name: the name of the theory
        params: a dictionary with the settable parameters of the model and their respective values
        metadata: a VariableCollection describing the variables of the model
        domain: a function which returns all the available X values for the model
        experiment_runner: a function which takes X values and returns simulated y values **with
            statistical noise**
        ground_truth: a function which takes X values and returns simulated y values **without any
            statistical noise**
        plotter: a function which plots the ground truth and, optionally, a model with a
            `predict` method (e.g. scikit-learn estimators)
    """

    name: Optional[str] = None
    params: Optional[Dict] = None
    metadata: Optional[VariableCollection] = None
    domain: Optional[Callable] = None
    experiment_runner: Optional[Callable] = None
    ground_truth: Optional[Callable] = None
    plotter: Optional[Callable[[Optional[SupportsPredict]], None]] = None


Inventory: Dict[str, SyntheticExperimentClosure] = dict()
""" The dictionary of `SyntheticExperimentCollection`. """


def register(id: str, closure: SyntheticExperimentClosure) -> None:
    """
    Add a new synthetic experiment to the Inventory.

    Parameters:
         id: the unique id for the model.
         closure: a function which returns a SyntheticExperimentCollection

    """
    Inventory[id] = closure


def retrieve(id: str, **kwargs) -> SyntheticExperimentCollection:
    """Retrieve a synthetic experiment from the Inventory.

    Parameters:
        id: the unique id for the model
        **kwargs: keyword arguments for the synthetic experiment (metadata, coefficients etc.)
    """
    closure: SyntheticExperimentClosure = Inventory[id]
    evaluated_closure = closure(**kwargs)
    return evaluated_closure
