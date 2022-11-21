"""
Provides tools to chain functions used to create experiment sequences.
"""
from typing import Any, Iterable, Protocol, Sequence, Union, runtime_checkable

import numpy as np


class ExperimentalCondition:
    """An ExperimentalCondition represents a trial."""

    pass


ExperimentalSequence = Iterable[ExperimentalCondition]
ExperimentalSequence.__doc__ = """
An ExperimentalSequence represents a series of trials.
"""


def sequence_to_ndarray(sequence: ExperimentalSequence):
    """Converts an ExperimentalSequence to a numpy-ndarray."""
    ndarray = np.ndarray(list(sequence))
    return ndarray


@runtime_checkable
class Pool(Protocol):
    """Creates an experimental sequence from scratch."""

    def __call__(self) -> ExperimentalSequence:
        ...


@runtime_checkable
class Pipe(Protocol):
    """Takes in an ExperimentalSequence and modifies it before returning it."""

    def __call__(self, ex: ExperimentalSequence) -> ExperimentalSequence:
        ...


class PoolPipeline:
    """
    Creates ("pools") and processes ("pipelines") a series of ExperimentalSequences.

    Examples:
        A pipeline which generates the list of integers from 0 to 9
        >>> p = PoolPipeline(
        ... lambda: range(10), # the "pool" function
        ... )
        >>> list(p())
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        A pipeline which generates the list of even values from 0 to 9
        >>> q = PoolPipeline(
        ... lambda: range(10),                                   # the "pool" function
        ... lambda values: filter(lambda i: i % 2 == 0, values)  # a "pipe" function
        ... )
        >>> list(q())
        [0, 2, 4, 6, 8]

        A pipeline which generates the list of even values from 0 to 4 paired with "a" and "b"
        >>> from itertools import product
        >>> r = PoolPipeline(
        ... lambda: product(range(5), ["a", "b"]),          # the "pool" function
        ... lambda values: filter(lambda i: i[0] % 2 == 0, values)  # a "pipe" function
        ... )
        >>> list(r())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b')]

        The pipeline can also be called using the "run" syntax:
        >>> list(r.run())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b')]

        If the pipeline is evaluated without wrapping the result in a function like "list" which
        ensures that all the values are really instantiated, it remains unevaluated.
        >>> s = r.run()
        >>> s # doctest: +ELLIPSIS
        <filter object at 0x...>

        The filter can be evaluated by wrapping it in a function like "list" as above, or "tuple":
        >>> tuple(s)
        ((0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b'))


    """

    def __init__(
        self, pool: Union[Pool, Iterable], *pipes: Pipe, params: dict[str, Any]
    ):
        """Initialize the pipeline with a Pool object and a series of Pipe objects."""
        self.pool = pool
        self.pipes = pipes
        self.params = params

    def __call__(self) -> ExperimentalSequence:
        """Create the pool of values, then successively pass it through the Pipe."""
        # Create pool
        results = []
        if isinstance(self.pool, Iterable):
            results = [self.pool]
        elif isinstance(self.pool, Pool):
            results = [self.pool()]

        # Run filters
        for pipe in self.pipes:
            results.append(pipe(results[-1]))

        return results[-1]

    run = __call__


def _parse_params_to_nested_dict(params_dict):
    nested_dictionary = {}
    for key, value in params_dict.items():
        part0, part1 = key.split("__", 1)
        subdictionary = nested_dictionary.get(part0, {})
        subdictionary.update({part1: value})
        nested_dictionary[part0] = subdictionary

    return nested_dictionary


def make_pipeline(steps: Sequence[Union[Pool, Iterable, Pipe]], params: dict[str, Any]):
    pool = steps[0]
    assert isinstance(pool, Pool) or isinstance(pool, Iterable)

    pipes = steps[1:]
    assert all([isinstance(pipe, Pipe) for pipe in pipes])

    pipeline = PoolPipeline(pool, *pipes, params=params)  # type: ignore ## todo: fix this
    return pipeline
