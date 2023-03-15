"""
Classes for storing and passing a cycle's state between Executors.

We provide two views of a cycle's state:
- by "history" – first datum, second datum ... last datum – where the results are strictly
  sequential.
- by "kind" – metadata, parameter, condition, observation, theory

Our fundamental representation is as a list of Results objects: the order in the list represents
history and the Result object holds both the data and metadata like the "kind".

Examples:
    We start with an emtpy history
    >>> from autora.cycle._state import history_to_kind, ResultKind, Result
    >>> history_ = []

    The view of this empty history on the "kind" dimension is also empty:
    >>> history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    namespace(conditions=[], metadata=VariableCollection(...), observations=[], params={},
              theories=[])

    We can add new results to the history:
    >>> history_.append(Result([1,2,3], ResultKind.CONDITION))

    ... and view the results:
    >>> history_to_kind(history_) # doctest: +ELLIPSIS
    namespace(conditions=[[1, 2, 3]], ...)

"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.utils.dictionary import LazyDict
from autora.variable import VariableCollection


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[Any]


class SupportsData(Protocol):
    """Object with an attribute which has a sequence of arbitrary data."""

    data: Sequence[SupportsDataKind]


class SupportsResults(Protocol):
    """Object with an attribute which has a sequence of conditions, observations and theories."""

    results: Sequence[SupportsDataKind]


def history_to_kind(history: Sequence[Result]):
    """
    Convert a sequence of results into a SimpleNamespace with attributes:
    - `.metadata`
    - `.params`
    - `.conditions`
    - `.observations`
    - `.theories`

    Examples:
        History might be empty
        >>> history_ = []
        >>> history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE
        namespace(conditions=[], metadata=VariableCollection(independent_variables=[],
            dependent_variables=[], covariates=[]), observations=[], params={}, theories=[])

        ... or with values for any or all of the parameters:
        >>> history_ = init_result_list(params={"some": "params"})
        >>> history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        namespace(... params={'some': 'params'}, ...)

        >>> history_ += init_result_list(conditions=["a condition"])
        >>> history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        namespace(conditions=['a condition'], ..., params={'some': 'params'}, ...)

        >>> history_to_kind(history_).params
        {'some': 'params'}

        >>> history_ += init_result_list(observations=["an observation"])
        >>> history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        namespace(conditions=['a condition'], ..., observations=['an observation'],
            params={'some': 'params'}, ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Result(LinearRegression(), kind=ResultKind.THEORY)]
        >>> history_to_kind(history_) # doctest: +ELLIPSIS
        namespace(..., theories=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> metadata = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Result(metadata, kind=ResultKind.METADATA)]
        >>> history_to_kind(history_) # doctest: +ELLIPSIS
        namespace(... metadata=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Result({'some': 'params'}, kind=ResultKind.PARAMS)]
        >>> history_to_kind(history_) # doctest: +ELLIPSIS
        namespace(..., params={'some': 'params'}, ...)

    """
    namespace = SimpleNamespace(
        metadata=get_last_data_with_default(
            history, kind={ResultKind.METADATA}, default=VariableCollection()
        ),
        params=get_last_data_with_default(
            history, kind={ResultKind.PARAMS}, default={}
        ),
        observations=list_data(filter_result(history, kind={ResultKind.OBSERVATION})),
        theories=list_data(filter_result(history, kind={ResultKind.THEORY})),
        conditions=list_data(filter_result(history, kind={ResultKind.CONDITION})),
    )
    return namespace


@dataclass(frozen=True)
class Result(SupportsDataKind):
    """
    Container class for data and metadata.
    """

    data: Optional[Any]
    kind: Optional[ResultKind]

    def __post_init__(self):
        object.__setattr__(self, "kind", ResultKind(self.kind))


class ResultKind(str, Enum):
    """
    Kinds of results which can be held in the Result object.

    Examples:
        >>> ResultKind.CONDITION is ResultKind.CONDITION
        True

        >>> ResultKind.CONDITION is ResultKind.METADATA
        False

        >>> ResultKind.CONDITION == "CONDITION"
        True

        >>> ResultKind.CONDITION == "METADATA"
        False

        >>> ResultKind.CONDITION in {ResultKind.CONDITION, ResultKind.PARAMS}
        True

        >>> ResultKind.METADATA in {ResultKind.CONDITION, ResultKind.PARAMS}
        False
    """

    CONDITION = "CONDITION"
    OBSERVATION = "OBSERVATION"
    THEORY = "THEORY"
    PARAMS = "PARAMS"
    METADATA = "METADATA"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


def list_data(data: Sequence[SupportsDataKind]):
    return list(r.data for r in data)


def filter_result(data: Iterable[SupportsDataKind], kind: Set[ResultKind]):
    return filter(lambda r: r.kind in kind, data)


def get_last(data: Sequence[SupportsDataKind], kind: Set[ResultKind]):
    results_new_to_old = reversed(data)
    last_of_kind = next(filter_result(results_new_to_old, kind=kind))
    return last_of_kind


def get_last_data_with_default(data: Sequence[SupportsDataKind], kind, default):
    try:
        result = get_last(data, kind).data
    except StopIteration:
        result = default
    return result


def init_result_list(
    metadata: Optional[VariableCollection] = None,
    params: Optional[Dict] = None,
    conditions: Optional[Iterable[ArrayLike]] = None,
    observations: Optional[Iterable[ArrayLike]] = None,
    theories: Optional[Iterable[BaseEstimator]] = None,
) -> List[Result]:
    """
    Initialize a list of Result objects

    Returns:

    Args:
        metadata: a single datum to be marked as "metadata"
        params: a single datum to be marked as "params"
        conditions: an iterable of data, each to be marked as "conditions"
        observations: an iterable of data, each to be marked as "observations"
        theories: an iterable of data, each to be marked as "theories"

    Examples:
        CycleState can be initialized in an empty state:
        >>> init_result_list()
        []

        ... or with values for any or all of the parameters:
        >>> from autora.variable import VariableCollection
        >>> init_result_list(metadata=VariableCollection()) # doctest: +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.METADATA)]

        >>> init_result_list(params={"some": "params"})
        [Result(data={'some': 'params'}, kind=ResultKind.PARAMS)]

        >>> init_result_list(conditions=["a condition"])
        [Result(data='a condition', kind=ResultKind.CONDITION)]

        >>> init_result_list(observations=["an observation"])
        [Result(data='an observation', kind=ResultKind.OBSERVATION)]

        >>> from sklearn.linear_model import LinearRegression
        >>> init_result_list(theories=[LinearRegression()])
        [Result(data=LinearRegression(), kind=ResultKind.THEORY)]

        The input arguments are added to the data in the order `metadata`,
        `params`, `conditions`, `observations`, `theories`:
        >>> init_result_list(metadata=VariableCollection(),
        ...                  params={"some": "params"},
        ...                  conditions=["a condition"],
        ...                  observations=["an observation", "another observation"],
        ...                  theories=[LinearRegression()],
        ... ) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.METADATA),
         Result(data={'some': 'params'}, kind=ResultKind.PARAMS),
         Result(data='a condition', kind=ResultKind.CONDITION),
         Result(data='an observation', kind=ResultKind.OBSERVATION),
         Result(data='another observation', kind=ResultKind.OBSERVATION),
         Result(data=LinearRegression(), kind=ResultKind.THEORY)]

    """
    data = []

    if metadata is not None:
        data.append(Result(metadata, ResultKind.METADATA))

    if params is not None:
        data.append(Result(params, ResultKind.PARAMS))

    for seq, kind in [
        (conditions, ResultKind.CONDITION),
        (observations, ResultKind.OBSERVATION),
        (theories, ResultKind.THEORY),
    ]:
        if seq is not None:
            for i in seq:
                data.append(Result(i, kind=kind))

    return data


def _resolve_state_params(state: Sequence[Result]) -> Dict:
    """
    Returns the `params` attribute of the input, with `cycle properties` resolved.

    Examples:
        >>> from autora.cycle._state import init_result_list, _resolve_state_params
        >>> s = init_result_list(theories=["the first theory", "the second theory"],
        ...     params={"experimentalist": {"source": "%theories[-1]%"}})
        >>> _resolve_state_params(s)
        {'experimentalist': {'source': 'the second theory'}}

    """
    state_dependent_properties = _get_state_dependent_properties(state)
    namespace_params = history_to_kind(state).params
    resolved_params = _resolve_properties(namespace_params, state_dependent_properties)
    return resolved_params


def _get_state_dependent_properties(state: Sequence[Result]):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> from autora.variable import VariableCollection
        >>> state_dependent_properties = _get_state_dependent_properties([])

        ... but it will raise an exception if a value isn't yet available when we try to use it
        >>> state_dependent_properties["%theories[-1]%"] # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: list index out of range

        Nevertheless, we can iterate through its keys no problem:
        >>> [key for key in state_dependent_properties.keys()] # doctest: +NORMALIZE_WHITESPACE
        ['%observations.ivs[-1]%', '%observations.dvs[-1]%', '%observations.ivs%',
        '%observations.dvs%', '%theories[-1]%', '%theories%']

    """
    namespace_view = history_to_kind(state)

    n_ivs = len(namespace_view.metadata.independent_variables)
    n_dvs = len(namespace_view.metadata.dependent_variables)
    state_dependent_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: namespace_view.observations[-1][
                :, 0:n_ivs
            ],
            "%observations.dvs[-1]%": lambda: namespace_view.observations[-1][
                :, n_ivs:
            ],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + namespace_view.observations
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(namespace_view.observations)[
                :, n_ivs:
            ],
            "%theories[-1]%": lambda: namespace_view.theories[-1],
            "%theories%": lambda: namespace_view.theories,
        }
    )
    return state_dependent_property_dict


def _resolve_properties(params: Dict, state_dependent_properties: Mapping):
    """
    Resolve state-dependent properties inside a nested dictionary.

    In this context, a state-dependent-property is a string which is meant to be replaced by its
    updated, current value before the dictionary is used. A state-dependent property might be
    something like "the last theorist available" or "all the experimental results until now".

    Args:
        params: a (nested) dictionary of keys and values, where some values might be
            "cycle property names"
        state_dependent_properties: a dictionary of "property names" and their "real values"

    Returns: a (nested) dictionary where "property names" are replaced by the "real values"

    Examples:

        >>> params_0 = {"key": "%foo%"}
        >>> cycle_properties_0 = {"%foo%": 180}
        >>> _resolve_properties(params_0,cycle_properties_0)
        {'key': 180}

        >>> params_1 = {"key": "%bar%", "nested_dict": {"inner_key": "%foobar%"}}
        >>> cycle_properties_1 = {"%bar%": 1, "%foobar%": 2}
        >>> _resolve_properties(params_1,cycle_properties_1)
        {'key': 1, 'nested_dict': {'inner_key': 2}}

    """
    params_ = copy.copy(params)
    for key, value in params_.items():
        if isinstance(value, dict):
            params_[key] = _resolve_properties(value, state_dependent_properties)
        elif (
            isinstance(value, str) and value in state_dependent_properties
        ):  # value is a key in the cycle_properties dictionary
            params_[key] = state_dependent_properties[value]
        else:
            pass  # no change needed

    return params_
