"""
Random experimentalist module.
"""
from typing import Callable, Generator, Sequence

import numpy as np
import numpy.random

from autora.variable import ValueType, Variable, VariableCollection


def experiment_proposer(
    x: np.ndarray,
    y: np.ndarray,
    variable_metadata: VariableCollection,
    theory: Callable[[np.ndarray], np.ndarray],
    rng: numpy.random.Generator,
) -> Generator[np.ndarray, None, None]:
    """
    Creates new experimental conditions based on a uniform random sample of
    the independent variable domain.

    Args:
        x: existing independent variable values for the theory
        y: known dependent variable values corresponding to the x
        metadata: a description of the variable domain
        theory: a function which represents the current best mapping of x -> y
        rng: the random number generator to use

    Returns:
        a generator which can be queried for new experimental conditions

    Examples:
        We can query the random proposer using the usual python iterable mechanisms:
        >>> proposer = experiment_proposer(
        ...     x=None, y=None, theory=None,
        ...     variable_metadata=VariableCollection(
        ...         independent_variables=[Variable(type="real", value_range=[5,10])]
        ...     ),
        ...     rng=np.random.default_rng(42)
        ... )
        >>> next(proposer)
        array([8.86978024])
        >>> next(proposer)
        array([7.1943922])

        The same applies if we've got two independent variables. Note that the values are
        repeated across runs if we re-seed the random number generator identically.
        >>> proposer = experiment_proposer(
        ...     x=None, y=None, theory=None,
        ...     variable_metadata=VariableCollection(
        ...         independent_variables=[
        ...             Variable(type="real", value_range=[5,10]),
        ...             Variable(type="real", value_range=[5,10])
        ...         ]
        ...     ),
        ...     rng=np.random.default_rng(42)
        ... )
        >>> next(proposer)
        array([8.86978024, 7.1943922 ])
        >>> next(proposer)
        array([9.2929896 , 8.48684015])

    """
    while True:
        x_prime = sample_randomly_from_variable_domains(
            variable_metadata.independent_variables, rng=rng
        )
        yield x_prime


def sample_randomly_from_variable_domains(
    variable_metadata: Sequence[Variable],
    rng: numpy.random.Generator,
) -> np.ndarray:
    """

    Args:
        variable_metadata: a description of the variable domains
        rng: the random number generator instance to use for sampling

    Returns:
        a single array of new x-values uniformly sampled from the variable domains

    Examples:
        To sample a single real-valued independent variable:
        >>> sample_randomly_from_variable_domains(
        ...     variable_metadata=[
        ...         Variable(type="real", value_range=[5,10]),
        ...     ],
        ...     rng=np.random.default_rng(42)
        ... )
        array([8.86978024])

        To sample multiple real-valued independent variables:
        >>> sample_randomly_from_variable_domains(
        ...     variable_metadata=[
        ...         Variable(type="real", value_range=[5,10]),
        ...         Variable(type="real", value_range=[15,32])
        ...     ],
        ...     rng=np.random.default_rng(42)
        ... )
        array([ 8.86978024, 22.46093348])

        Other value types will throw an error:
        >>> sample_randomly_from_variable_domains(
        ...     variable_metadata=[
        ...         Variable(type="sigmoid"),
        ...     ],
        ...     rng=np.random.default_rng(42)
        ... )
        Traceback (most recent call last):
        ...
        NotImplementedError: value type sigmoid cannot yet be sampled


    """
    x_prime_list = []
    for x_meta in variable_metadata:
        if x_meta.type == ValueType.REAL:
            x_prime_list.append(rng.uniform(low=x_meta.min, high=x_meta.max))
        else:
            raise NotImplementedError(f"value type {x_meta.type} cannot yet be sampled")
    x_prime = np.asarray(x_prime_list)
    return x_prime
