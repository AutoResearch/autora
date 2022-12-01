"""
Provides tools to chain functions used to create experiment sequences.
"""
from __future__ import annotations

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


@runtime_checkable
class Pool(Protocol):
    """Creates an experimental sequence from scratch."""

    def __call__(self) -> _ExperimentalSequence:
        ...


@runtime_checkable
class Pipe(Protocol):
    """Takes in an _ExperimentalSequence and modifies it before returning it."""

    def __call__(self, ex: _ExperimentalSequence) -> _ExperimentalSequence:
        ...


_StepType = Tuple[str, Union[Pool, Pipe, Iterable]]
_StepType.__doc__ = (
    "A Pipeline step's name and generating object, as tuple(name, pipeline_piece)."
)

PARAM_DIVIDER = "__"


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
        Pipeline(steps=[('pool', <function <lambda> at 0x...>)], params={})

        >>> Pipeline([
        ... ("pool", lambda: product(range(5), ["a", "b"])),
        ... ("filter", lambda values: filter(lambda i: i[0] % 2 == 0, values))
        ... ]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('pool', <function <lambda> at 0x...>), \
        ('filter', <function <lambda> at 0x...>)], \
        params={})

        >>> pipeline = Pipeline([
        ... ("pool", lambda maximum: product(range(maximum), ["a", "b"])),
        ... ("filter", lambda values, divisor: filter(lambda i: i[0] % divisor == 0, values))
        ... ] ,
        ... params = {"pool": {"maximum":5}, "filter": {"divisor": 2}})
        >>> pipeline # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('pool', <function <lambda> at 0x...>), \
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
        steps: Optional[Sequence[_StepType]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the pipeline with a series of Pipe objects."""
        if steps is None:
            steps = list()
        self.steps = steps

        if params is None:
            params = dict()
        self.params = params

    def __repr__(self):
        return f"Pipeline(steps={self.steps}, params={self.params})"

    def __call__(
        self,
        ex: Optional[_ExperimentalSequence] = None,
        **params,
    ) -> _ExperimentalSequence:
        """Successively pass the input values through the Pipe."""

        # Initialize the parameters objects.
        pipeline_params = _parse_params_to_nested_dict(
            self.params, divider=PARAM_DIVIDER
        )
        call_params = _parse_params_to_nested_dict(params, divider=PARAM_DIVIDER)
        merged_params = _merge_dicts(pipeline_params, call_params)

        try:
            # Check we have steps to use
            assert len(self.steps) > 0
        except AssertionError:
            # If the pipeline doesn't have any steps...
            if ex is not None:
                # ...the output is the input
                return ex
            elif ex is None:
                # ... unless the input was None, in which case it's an emtpy list
                return []

        # Make an iterator from the steps, so that we can be sure to only go through them once
        # (Otherwise if we handle the "pool" as a special case, we have to track our starting point)
        pipes_iterator = iter(self.steps)

        # Initialize our results object
        if ex is None:
            # ... there's no input, so presumably the first element in the steps is a pool
            # which should generate our initial values.
            name, pool = next(pipes_iterator)
            if isinstance(pool, Pool):
                # Here, the pool is a Pool callable, which we can pass parameters.
                all_params_for_pool = merged_params.get(name, dict())
                results = [pool(**all_params_for_pool)]
            elif isinstance(pool, Iterable):
                # Otherwise, the pool should be an iterable which we can just use as is.
                results = [pool]

        else:
            # ... there's some input, so we can use that as the initial value
            results = [ex]

        # Run the successive steps over the last result
        for name, pipe in pipes_iterator:
            assert isinstance(pipe, Pipe)
            all_params_for_pipe = merged_params.get(name, dict())
            results.append(pipe(results[-1], **all_params_for_pipe))

        return results[-1]

    run = __call__


def _merge_dicts(a: dict, b: dict):
    """
    merges b into a.

    Args:
        a: the "base" dictionary
        b: the "update" dictionary which takes precendence

    Returns:

    Originally from https://stackoverflow.com/a/7205107, modified for AER to allow overwriting.

    Examples:
        Non-conflicting dictionaries are merged "side-by-side"
        >>> _merge_dicts({1:{"a":"A"},2:{"b":"B"}}, {2:{"c":"C"},3:{"d":"D"}})
        {1: {'a': 'A'}, 2: {'b': 'B', 'c': 'C'}, 3: {'d': 'D'}}

        With conflicting dictionaries, the second dictionary takes precedence
        >>> _merge_dicts(
        ...     {"l1_a": {"l2_1": {"l3_alpha": "from_first"}}},
        ...     {"l1_a": {"l2_1": {"l3_alpha": "from_second"}}})
        {'l1_a': {'l2_1': {'l3_alpha': 'from_second'}}}

        Again, with non-conflicting dictionaries at the lower level
        >>> _merge_dicts(
        ...     {"l1_a": {"l2_1": {"l3_alpha": "from_first"}}},
        ...     {"l1_a": {"l2_1": {"l3_beta": "from_second"}}})
        {'l1_a': {'l2_1': {'l3_alpha': 'from_first', 'l3_beta': 'from_second'}}}

        >>> _merge_dicts(
        ...     {"l1_a": {"l2_1": {"l3_alpha": "from_first", "l3_beta": "from_first"}}},
        ...     {"l1_a": {"l2_1": {                          "l3_beta": "from_second"}}})
        {'l1_a': {'l2_1': {'l3_alpha': 'from_first', 'l3_beta': 'from_second'}}}

    """
    a_, b_ = dict(a), dict(b)

    for key in b_:
        if key in a_:
            if isinstance(a_[key], dict) and isinstance(b_[key], dict):
                a_[key] = _merge_dicts(a_[key], b_[key])
            elif a_[key] != b_[key]:
                a_[key] = b_[key]
            else:
                pass
        else:
            a_[key] = b_[key]
    return a_


def _parse_params_to_nested_dict(params_dict: Dict, divider: str):
    """
    Converts a dictionary with a single level to a multi-level nested dictionary.

    Examples:
        >>> _parse_params_to_nested_dict({"a": 1}, divider="__")
        {'a': 1}
        >>> _parse_params_to_nested_dict({"a__b": 1, "a__c": 2}, divider="__")
        {'a': {'b': 1, 'c': 2}}
        >>> _parse_params_to_nested_dict(
        ...     {"a__b__alpha": 1, "a__b__beta": 2, "a__c__gamma": 3},
        ...     divider="__")
        {'a': {'b': {'alpha': 1, 'beta': 2}, 'c': {'gamma': 3}}}

        >>> _parse_params_to_nested_dict(
        ...     {"a:b:alpha": 1, "a:b:beta": 2, "a:c:gamma": 3},
        ...     divider=":")
        {'a': {'b': {'alpha': 1, 'beta': 2}, 'c': {'gamma': 3}}}
    """
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
        Pipeline(steps=[('<lambda>', <function <lambda> at 0x...>)], params={})

        You can create pipelines with normal functions.
        >>> def ab_pool(maximum=5): return product(range(maximum), ["a", "b"])
        >>> def even_filter(values): return filter(lambda i: i[0] % 2 == 0, values)
        >>> make_pipeline([ab_pool, even_filter]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('ab_pool', <function ab_pool at 0x...>), \
        ('even_filter', <function even_filter at 0x...>)], params={})

        You can create pipelines with generators as their first elements functions.
        >>> ab_pool_gen = product(range(3), ["a", "b"])
        >>> pl = make_pipeline([ab_pool_gen, even_filter])
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('step', <itertools.product object at 0x...>),
        ('even_filter', <function even_filter at 0x...>)], params={})
        >>> list(pl.run())
        [(0, 'a'), (0, 'b'), (2, 'a'), (2, 'b')]

        You can pass parameters into the different steps of the pl using the "params"
        argument:
        >>> def divisor_filter(x, divisor): return filter(lambda i: i[0] % divisor == 0, x)
        >>> pl = make_pipeline([ab_pool, divisor_filter],
        ... params = {"ab_pool": {"maximum":5}, "divisor_filter": {"divisor": 2}})
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('ab_pool', <function ab_pool at 0x...>), \
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
        Pipeline(steps=[('step', functools.partial(<function ab_pool at 0x...>, maximum=100))], \
        params={})

        If there are multiple steps with the same name, they get suffixes as usual:
        >>> pl = make_pipeline([partial(range, stop=10), partial(divisor_filter, divisor=3)])
        >>> pl # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Pipeline(steps=[('step_0', functools.partial(<class 'range'>, stop=10)), \
        ('step_1', functools.partial(<function divisor_filter at 0x...>, divisor=3))], \
        params={})



    """

    if steps is None:
        steps = []
    steps_: List[_StepType] = []
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


class _ExperimentalCondition:
    """An _ExperimentalCondition represents a trial."""

    pass


_ExperimentalSequence = Iterable[_ExperimentalCondition]
_ExperimentalSequence.__doc__ = """
An _ExperimentalSequence represents a series of trials.
"""
