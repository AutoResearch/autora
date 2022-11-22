"""
Provides tools to chain functions used to create experiment sequences.
"""
import copy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    get_args,
    runtime_checkable,
)

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


StepType = Tuple[str, Union[Pool, Pipe, Iterable]]
StepType.__doc__ = (
    "A Pipeline step's name and generating object, as tuple(name, object)."
)


class Pipeline:
    """
    Processes ("pipelines") a series of ExperimentalSequences through a pipeline.

    Examples:
        A pipeline which filters even values 0 to 9:
        >>> p = Pipeline(
        ... [("is_even", lambda values: filter(lambda i: i % 2 == 0, values))]  # a "pipe" function
        ... )
        >>> list(p(range(10)))
        [0, 2, 4, 6, 8]

        A pipeline which filters for square, odd numbers:
        >>> from math import sqrt
        >>> p = Pipeline([
        ... ("is_odd", lambda values: filter(lambda i: i % 2 != 0, values)),
        ... ("is_sqrt", lambda values: filter(lambda i: sqrt(i) % 1 == 0., values))
        ... ])
        >>> list(p(range(100)))
        [1, 9, 25, 49, 81]


        >>> from itertools import product
        >>> Pipeline([("pool", lambda: product(range(5), ["a", "b"]))]) # doctest: +ELLIPSIS
        Pipeline(pipes=[('pool', <function <lambda> at 0x...>)], params={})

        >>> Pipeline([
        ... ("pool", lambda: product(range(5), ["a", "b"])),
        ... ("filter", lambda values: filter(lambda i: i[0] % 2 == 0, values))
        ... ]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('pool', <function <lambda> at 0x...>), \
        ('filter', <function <lambda> at 0x...>)], \
        params={})

        >>> pipeline = Pipeline([
        ... ("pool", lambda maximum: product(range(maximum), ["a", "b"])),
        ... ("filter", lambda values, divisor: filter(lambda i: i[0] % divisor == 0, values))
        ... ] ,
        ... params = {"pool": {"maximum":5}, "filter": {"divisor": 2}})
        >>> pipeline # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('pool', <function <lambda> at 0x...>), \
        ('filter', <function <lambda> at 0x...>)], \
        params={'pool': {'maximum': 5}, 'filter': {'divisor': 2}})
        >>> list(pipeline.run())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b')]

        >>> pipeline.params = {"pool": {"maximum":7}, "filter": {"divisor": 3}}
        >>> list(pipeline())
        [(0, 'a'), (0, 'b'), (3, 'a'), (3, 'b'), (6, 'a'), (6, 'b')]

        >>> pipeline.params = {"pool": {"maximum":7}}
        >>> list(pipeline()) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: <lambda>() missing 1 required positional argument: 'divisor'


    """

    def __init__(
        self,
        pipes: Optional[Sequence[StepType]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the pipeline with a series of Pipe objects."""
        if pipes is None:
            pipes = list()
        self.pipes = pipes

        if params is None:
            params = dict()
        self.params = params

    def __repr__(self):
        return f"Pipeline(pipes={self.pipes}, params={self.params})"

    def __call__(
        self,
        ex: Optional[ExperimentalSequence] = None,
        **params,
    ) -> ExperimentalSequence:
        """Successively pass the input values through the Pipe."""

        # Initialize the parameters objects.
        pipeline_params = _parse_params_to_nested_dict(self.params)
        call_params = _parse_params_to_nested_dict(params)

        try:
            # Check we have pipes to use
            assert len(self.pipes) > 0
        except AssertionError:
            # If the pipeline doesn't have any pipes...
            if ex is not None:
                # ...the output is the input
                return ex
            elif ex is None:
                # ... unless the input was None, in which case it's an emtpy list
                return []

        # Make an iterator from the pipes, so that we can be sure to only go through them once
        # (Otherwise if we handle the "pool" as a special case, we have to track our starting point)
        pipes_iterator = iter(self.pipes)

        # Initialize our results object
        if ex is None:
            # ... there's no input, so presumably the first element in the pipes is a pool
            # which should generate our initial values.
            name, pool = next(pipes_iterator)
            if isinstance(pool, Pool):
                # Here, the pool is a Pool callable, which we can pass parameters.
                all_params_for_pool = self._get_params_for_name(
                    pipeline_params, call_params, name
                )
                results = [pool(**all_params_for_pool)]
            elif isinstance(pool, Iterable):
                # Otherwise, the pool should be an iterable which we can just use as is.
                results = [pool]

        else:
            # ... there's some input, so we can use that as the initial value
            results = [ex]

        # Run the successive pipes over the last result
        for name, pipe in pipes_iterator:
            assert isinstance(pipe, Pipe)
            all_params_for_pipe = self._get_params_for_name(
                pipeline_params, call_params, name
            )
            results.append(pipe(results[-1], **all_params_for_pipe))

        return results[-1]

    @staticmethod
    def _get_params_for_name(pipeline_params, call_params, name):
        pipeline_params_for_pipe = pipeline_params.get(name, dict())
        call_params_for_pipe = call_params.get(name, dict())
        all_params_for_pipe = dict(pipeline_params_for_pipe, **call_params_for_pipe)
        return all_params_for_pipe

    run = __call__


def _parse_params_to_nested_dict(params_dict: Dict, divider: str = "__"):
    nested_dictionary: dict = copy.copy(params_dict)
    for key in params_dict.keys():
        if divider in key:
            value = nested_dictionary.pop(key)
            new_key, new_subkey = key.split(divider, 1)
            subdictionary = nested_dictionary.get(new_key, {})
            subdictionary.update({new_subkey: value})
            nested_dictionary[new_key] = subdictionary

    for key, value in nested_dictionary.items():
        if isinstance(value, dict):
            nested_dictionary[key] = _parse_params_to_nested_dict(
                value, divider=divider
            )

    return nested_dictionary


def make_pipeline(
    steps: Optional[Sequence[Union[Pool, Pipe]]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    A factory function to make pipeline objects.

    The pipe objects' names will be set to the lowercase of their types, plus an index
    starting from 0 for non-unique names.

    Args:
        steps: a sequence of Pipe-compatible objects
        params: a dictionary of parameters passed to each Pipe by its inferred name

    Returns:
        A pipeline object

    Examples:

        You can create pipelines using purely anonymous functions:
        >>> from itertools import product
        >>> make_pipeline([lambda: product(range(5), ["a", "b"])]) # doctest: +ELLIPSIS
        Pipeline(pipes=[('<lambda>', <function <lambda> at 0x...>)], params={})

        You can create pipelines with normal functions.
        >>> def ab_pool(maximum=5): return product(range(maximum), ["a", "b"])
        >>> def even_filter(values): return filter(lambda i: i[0] % 2 == 0, values)
        >>> make_pipeline([ab_pool, even_filter]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('ab_pool', <function ab_pool at 0x...>), \
        ('even_filter', <function even_filter at 0x...>)], params={})

        You can create pipelines with generators as their first elements functions.
        >>> ab_pool_gen = product(range(3), ["a", "b"])
        >>> pl = make_pipeline([ab_pool_gen, even_filter])
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('step', <itertools.product object at 0x...>),
        ('even_filter', <function even_filter at 0x...>)], params={})
        >>> list(pl.run())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b')]

        You can pass parameters into the different steps of the pl using the "params"
        argument:
        >>> def divisor_filter(x, divisor): return filter(lambda i: i[0] % divisor == 0, x)
        >>> pl = make_pipeline([ab_pool, divisor_filter],
        ... params = {"ab_pool": {"maximum":5}, "divisor_filter": {"divisor": 2}})
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('ab_pool', <function ab_pool at 0x...>), \
        ('divisor_filter', <function divisor_filter at 0x...>)], \
        params={'ab_pool': {'maximum': 5}, 'divisor_filter': {'divisor': 2}})

        You can evaluate the pipeline means calling its `run` method:
        >>> list(pl.run())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b')]

        ... or calling it directly:
        >>> list(pl())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b'), (4, 'a'), (4, 'b')]

        You can update the parameters and evaluate again, giving different results:
        >>> pl.params = {"ab_pool": {"maximum": 7}, "divisor_filter": {"divisor": 3}}
        >>> list(pl())
        [(0, 'a'), (0, 'b'), (3, 'a'), (3, 'b'), (6, 'a'), (6, 'b')]

        If the pipeline needs parameters, then removing them will break the pipeline:
        >>> pl.params = {}
        >>> list(pl()) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: divisor_filter() missing 1 required positional argument: 'divisor'

        If multiple steps have the same inferred name, then they are given a suffix automatically,
        which has to be reflected in the params if used:
        >>> pl = make_pipeline([ab_pool, divisor_filter, divisor_filter])
        >>> pl.params = {
        ...     "ab_pool": {"maximum": 22},
        ...     "divisor_filter_0": {"divisor": 3},
        ...     "divisor_filter_1": {"divisor": 7}
        ... }
        >>> list(pl())
        [(0, 'a'), (0, 'b'), (21, 'a'), (21, 'b')]

        You can also use "partial" functions to include Pipes with defaults in the pipeline.
        Because the `partial` function doesn't inherit the __name__ of the original function,
        these steps are renamed to "step".
        >>> from functools import partial
        >>> pl = make_pipeline([partial(ab_pool, maximum=100)])
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('step', functools.partial(<function ab_pool at 0x...>, maximum=100))], \
        params={})

        If there are multiple steps with the same name, they get suffixes as usual:
        >>> pl = make_pipeline([partial(range, stop=10), partial(divisor_filter, divisor=3)])
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(pipes=[('step_0', functools.partial(<class 'range'>, stop=10)), \
        ('step_1', functools.partial(<function divisor_filter at 0x...>, divisor=3))], \
        params={})



    """

    if steps is None:
        steps = []
    steps_: List[StepType] = []
    raw_names_ = [getattr(pipe, "__name__", "step").lower() for pipe in steps]
    names_tally_ = dict([(name, raw_names_.count(name)) for name in set(raw_names_)])
    names_index_ = dict([(name, 0) for name in set(raw_names_)])

    for name, pipe in zip(raw_names_, steps):
        assert isinstance(pipe, get_args(Union[Pipe, Pool, Iterable]))

        if names_tally_[name] > 1:
            current_index_for_this_name = names_index_.get(name, 0)
            name_in_pipeline = f"{name}_{current_index_for_this_name}"
            names_index_[name] += 1
        else:
            name_in_pipeline = name

        steps_.append((name_in_pipeline, pipe))

    pipeline = Pipeline(steps_, params=params)

    return pipeline
