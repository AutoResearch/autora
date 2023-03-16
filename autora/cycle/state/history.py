"""
Classes for storing and passing a cycle's state, including its history.

We provide two views of a cycle's state:
- by "history" – first datum, second datum ... last datum – where the results are strictly
  sequential.
- by "kind" – metadata, parameter, condition, observation, theory
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.cycle.protocol.v1 import SupportsDataKind
from autora.cycle.state.simple import SimpleCycleData
from autora.utils.dictionary import LazyDict
from autora.variable import VariableCollection


class SimpleCycleDataHistory:
    """
    An immutable object for tracking the state and history of an AER cycle.
    """

    def __init__(
        self,
        metadata: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        conditions: Optional[List[ArrayLike]] = None,
        observations: Optional[List[ArrayLike]] = None,
        theories: Optional[List[BaseEstimator]] = None,
        history: Optional[Sequence[Result]] = None,
    ):
        """

        Args:
            metadata: a single datum to be marked as "metadata"
            params: a single datum to be marked as "params"
            conditions: an iterable of data, each to be marked as "conditions"
            observations: an iterable of data, each to be marked as "observations"
            theories: an iterable of data, each to be marked as "theories"
            history: an iterable of Result objects to be used as the initial history.

        Examples:
            Empty input leads to an empty state:
            >>> SimpleCycleDataHistory()
            SimpleCycleDataHistory([])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> SimpleCycleDataHistory(metadata=VariableCollection()) # doctest: +ELLIPSIS
            SimpleCycleDataHistory([Result(data=VariableCollection(...), kind=ResultKind.METADATA)])

            >>> SimpleCycleDataHistory(params={"some": "params"})
            SimpleCycleDataHistory([Result(data={'some': 'params'}, kind=ResultKind.PARAMS)])

            >>> SimpleCycleDataHistory(conditions=["a condition"])
            SimpleCycleDataHistory([Result(data='a condition', kind=ResultKind.CONDITION)])

            >>> SimpleCycleDataHistory(observations=["an observation"])
            SimpleCycleDataHistory([Result(data='an observation', kind=ResultKind.OBSERVATION)])

            >>> from sklearn.linear_model import LinearRegression
            >>> SimpleCycleDataHistory(theories=[LinearRegression()])
            SimpleCycleDataHistory([Result(data=LinearRegression(), kind=ResultKind.THEORY)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `metadata`, `params`, `conditions`, `observations`, `theories`
            >>> SimpleCycleDataHistory(theories=['t1', 't2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'}, metadata=VariableCollection(),
            ...     history=[Result("from history", ResultKind.METADATA)]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='from history', kind=ResultKind.METADATA),
                                    Result(data=VariableCollection(...), kind=ResultKind.METADATA),
                                    Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
                                    Result(data='c1', kind=ResultKind.CONDITION),
                                    Result(data='c2', kind=ResultKind.CONDITION),
                                    Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION),
                                    Result(data='t1', kind=ResultKind.THEORY),
                                    Result(data='t2', kind=ResultKind.THEORY)])
        """
        self._history: List

        if history is not None:
            self._history = list(history)
        else:
            self._history = []

        self._history += _init_result_list(
            metadata=metadata,
            params=params,
            conditions=conditions,
            observations=observations,
            theories=theories,
        )

    def update(
        self,
        metadata=None,
        params=None,
        conditions=None,
        observations=None,
        theories=None,
        history=None,
    ):
        """
        Create a new object with updated values.

        Examples:
            The initial object is empty:
            >>> s0 = SimpleCycleDataHistory()
            >>> s0
            SimpleCycleDataHistory([])

            We can update the metadata using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> s1 = s0.update(metadata=VariableCollection())
            >>> s1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data=VariableCollection(...), kind=ResultKind.METADATA)])

            ... the original object is unchanged:
            >>> s0
            SimpleCycleDataHistory([])

            We can update the metadata again:
            >>> s2 = s1.update(metadata=VariableCollection(["some IV"]))
            >>> s2._by_kind  # doctest: +ELLIPSIS
            SimpleCycleData(metadata=VariableCollection(independent_variables=['some IV'],...), ...)

            ... and we see that there is only ever one metadata object returned.

            Params is treated the same way as metadata:
            >>> sp = s0.update(params={'first': 'params'})
            >>> sp
            SimpleCycleDataHistory([Result(data={'first': 'params'}, kind=ResultKind.PARAMS)])

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> sp = sp.update(params={'second': 'params'})
            >>> sp.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> sp  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data={'first': 'params'}, kind=ResultKind.PARAMS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMS)])

            When we update the conditions, observations or theories, a new entry is added to the
            history:
            >>> s3 = s0.update(theories=["1st theory"])
            >>> s3  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='1st theory', kind=ResultKind.THEORY)])

            ... so we can see the history of all the theories, for instance.
            >>> s3 = s3.update(theories=["2nd theory"])  # doctest: +NORMALIZE_WHITESPACE
            >>> s3  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='1st theory', kind=ResultKind.THEORY),
                                    Result(data='2nd theory', kind=ResultKind.THEORY)])

            ... and the full history of theories is available using the `.theories` parameter:
            >>> s3.theories
            ['1st theory', '2nd theory']

            The same for the observations:
            >>> s4 = s0.update(observations=["1st observation"])
            >>> s4
            SimpleCycleDataHistory([Result(data='1st observation', kind=ResultKind.OBSERVATION)])

            >>> s4.update(observations=["2nd observation"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='1st observation', kind=ResultKind.OBSERVATION),
                                    Result(data='2nd observation', kind=ResultKind.OBSERVATION)])


            The same for the conditions:
            >>> s5 = s0.update(conditions=["1st condition"])
            >>> s5
            SimpleCycleDataHistory([Result(data='1st condition', kind=ResultKind.CONDITION)])

            >>> s5.update(conditions=["2nd condition"])  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='1st condition', kind=ResultKind.CONDITION),
                                    Result(data='2nd condition', kind=ResultKind.CONDITION)])

            You can also update with multiple conditions, observations and theories:
            >>> s0.update(conditions=['c1', 'c2'])  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data='c1', kind=ResultKind.CONDITION),
                                    Result(data='c2', kind=ResultKind.CONDITION)])

            >>> s0.update(theories=['t1', 't2'], metadata={'m': 1}) # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data={'m': 1}, kind=ResultKind.METADATA),
                                    Result(data='t1', kind=ResultKind.THEORY),
                                    Result(data='t2', kind=ResultKind.THEORY)])

            >>> s0.update(theories=['t1'], observations=['o1'], metadata={'m': 1}
            ... )  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data={'m': 1}, kind=ResultKind.METADATA),
                                    Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='t1', kind=ResultKind.THEORY)])

        """

        if history is not None:
            history_extension = history
        else:
            history_extension = []

        history_extension += _init_result_list(
            metadata=metadata,
            params=params,
            conditions=conditions,
            observations=observations,
            theories=theories,
        )
        new_full_history = self._history + history_extension

        return SimpleCycleDataHistory(history=new_full_history)

    def __repr__(self):
        return f"{type(self).__name__}({self.history})"

    @property
    def _by_kind(self):
        return _history_to_kind(self._history)

    @property
    def metadata(self) -> VariableCollection:
        """

        Examples:
            The initial object is empty:
            >>> s = SimpleCycleDataHistory()

            ... and returns an emtpy metadata object
            >>> s.metadata
            VariableCollection(independent_variables=[], dependent_variables=[], covariates=[])

            We can update the metadata using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> s = s.update(metadata=VariableCollection(independent_variables=['some IV']))
            >>> s.metadata  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some IV'], ...)

            We can update the metadata again:
            >>> s = s.update(metadata=VariableCollection(["some other IV"]))
            >>> s.metadata  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some other IV'], ...)

            ... and we see that there is only ever one metadata object returned."""
        return self._by_kind.metadata

    @property
    def params(self) -> Dict:
        """

        Returns:

        Examples:
            Params is treated the same way as metadata:
            >>> s = SimpleCycleDataHistory()
            >>> s = s.update(params={'first': 'params'})
            >>> s.params
            {'first': 'params'}

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> s = s.update(params={'second': 'params'})
            >>> s.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> s  # doctest: +NORMALIZE_WHITESPACE
            SimpleCycleDataHistory([Result(data={'first': 'params'}, kind=ResultKind.PARAMS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMS)])
        """
        return self._by_kind.params

    @property
    def conditions(self) -> List[ArrayLike]:
        """
        Returns:

        Examples:
            View the sequence of theories with one conditions:
            >>> s = SimpleCycleDataHistory(conditions=[(1,2,3,)])
            >>> s.conditions
            [(1, 2, 3)]

            ... or more conditions:
            >>> s = s.update(conditions=[(4,5,6),(7,8,9)])  # doctest: +NORMALIZE_WHITESPACE
            >>> s.conditions
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        """
        return self._by_kind.conditions

    @property
    def observations(self) -> List[ArrayLike]:
        """

        Returns:

        Examples:
            The sequence of all observations is returned
            >>> s = SimpleCycleDataHistory(observations=["1st observation"])
            >>> s.observations
            ['1st observation']

            >>> s = s.update(observations=["2nd observation"])
            >>> s.observations  # doctest: +ELLIPSIS
            ['1st observation', '2nd observation']

        """
        return self._by_kind.observations

    @property
    def theories(self) -> List[BaseEstimator]:
        """

        Returns:

        Examples:
            View the sequence of theories with one theory:
            >>> s = SimpleCycleDataHistory(theories=["1st theory"])
            >>> s.theories  # doctest: +NORMALIZE_WHITESPACE
            ['1st theory']

            ... or more theories:
            >>> s = s.update(theories=["2nd theory"])  # doctest: +NORMALIZE_WHITESPACE
            >>> s.theories
            ['1st theory', '2nd theory']

        """
        return self._by_kind.theories

    @property
    def history(self) -> List[Result]:
        """

        Examples:
            We initialze some history:
            >>> s = SimpleCycleDataHistory(theories=['t1', 't2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'}, metadata=VariableCollection(),
            ...     history=[Result("from history", ResultKind.METADATA)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `metadata`, `params`, `conditions`, `observations`, `theories`

            >>> s.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='from history', kind=ResultKind.METADATA),
             Result(data=VariableCollection(...), kind=ResultKind.METADATA),
             Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
             Result(data='c1', kind=ResultKind.CONDITION),
             Result(data='c2', kind=ResultKind.CONDITION),
             Result(data='o1', kind=ResultKind.OBSERVATION),
             Result(data='o2', kind=ResultKind.OBSERVATION),
             Result(data='t1', kind=ResultKind.THEORY),
             Result(data='t2', kind=ResultKind.THEORY)]

            If we add a new value, like the params object, the updated value is added to the
            end of the history:
            >>> s = s.update(params={'new': 'param'})
            >>> s.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [..., Result(data={'new': 'param'}, kind=ResultKind.PARAMS)]

            The history can be filtered using the `filter_result` function:
            >>> list(filter_history(s.history, {ResultKind.PARAMS})
            ... )  # doctest: +NORMALIZE_WHITESPACE
            [Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
             Result(data={'new': 'param'}, kind=ResultKind.PARAMS)]

        """
        return self._history


def _history_to_kind(history: Sequence[Result]) -> SimpleCycleData:
    """
    Convert a sequence of results into a SimpleCycleData instance:

    Examples:
        History might be empty
        >>> history_ = []
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        SimpleCycleData(metadata=VariableCollection(...), params={},
                        conditions=[], observations=[], theories=[])

        ... or with values for any or all of the parameters:
        >>> history_ = _init_result_list(params={"some": "params"})
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        SimpleCycleData(..., params={'some': 'params'}, ...)

        >>> history_ += _init_result_list(conditions=["a condition"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        SimpleCycleData(..., params={'some': 'params'}, conditions=['a condition'], ...)

        >>> _history_to_kind(history_).params
        {'some': 'params'}

        >>> history_ += _init_result_list(observations=["an observation"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        SimpleCycleData(..., params={'some': 'params'}, conditions=['a condition'],
                        observations=['an observation'], ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Result(LinearRegression(), kind=ResultKind.THEORY)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        SimpleCycleData(..., theories=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> metadata = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Result(metadata, kind=ResultKind.METADATA)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        SimpleCycleData(metadata=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Result({'some': 'params'}, kind=ResultKind.PARAMS)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        SimpleCycleData(..., params={'some': 'params'}, ...)

    """
    namespace = SimpleCycleData(
        metadata=_get_last_data_with_default(
            history, kind={ResultKind.METADATA}, default=VariableCollection()
        ),
        params=_get_last_data_with_default(
            history, kind={ResultKind.PARAMS}, default={}
        ),
        observations=_list_data(filter_history(history, kind={ResultKind.OBSERVATION})),
        theories=_list_data(filter_history(history, kind={ResultKind.THEORY})),
        conditions=_list_data(filter_history(history, kind={ResultKind.CONDITION})),
    )
    return namespace


@dataclass(frozen=True)
class Result(SupportsDataKind):
    """
    Container class for data and metadata.

    >>> Result()
    Result(data=None, kind=None)

    >>> Result("a")
    Result(data='a', kind=None)

    >>> Result(None, "THEORY")
    Result(data=None, kind=ResultKind.THEORY)

    >>> Result(data="b")
    Result(data='b', kind=None)

    >>> Result("c", "OBSERVATION")
    Result(data='c', kind=ResultKind.OBSERVATION)
    """

    data: Optional[Any] = None
    kind: Optional[ResultKind] = None

    def __post_init__(self):
        if isinstance(self.kind, str):
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


def _list_data(data: Sequence[SupportsDataKind]):
    """
    Extract the `.data` attribute of each item in a sequence, and return as a list.

    Examples:
        >>> _list_data([])
        []

        >>> _list_data([Result("a"), Result("b")])
        ['a', 'b']
    """
    return list(r.data for r in data)


def filter_history(data: Iterable[SupportsDataKind], kind: Set[ResultKind]):
    return filter(lambda r: r.kind in kind, data)


def _get_last(data: Sequence[SupportsDataKind], kind: Set[ResultKind]):
    results_new_to_old = reversed(data)
    last_of_kind = next(filter_history(results_new_to_old, kind=kind))
    return last_of_kind


def _get_last_data_with_default(data: Sequence[SupportsDataKind], kind, default):
    try:
        result = _get_last(data, kind).data
    except StopIteration:
        result = default
    return result


def _init_result_list(
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
        Empty input leads to an empty state:
        >>> _init_result_list()
        []

        ... or with values for any or all of the parameters:
        >>> from autora.variable import VariableCollection
        >>> _init_result_list(metadata=VariableCollection()) # doctest: +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.METADATA)]

        >>> _init_result_list(params={"some": "params"})
        [Result(data={'some': 'params'}, kind=ResultKind.PARAMS)]

        >>> _init_result_list(conditions=["a condition"])
        [Result(data='a condition', kind=ResultKind.CONDITION)]

        >>> _init_result_list(observations=["an observation"])
        [Result(data='an observation', kind=ResultKind.OBSERVATION)]

        >>> from sklearn.linear_model import LinearRegression
        >>> _init_result_list(theories=[LinearRegression()])
        [Result(data=LinearRegression(), kind=ResultKind.THEORY)]

        The input arguments are added to the data in the order `metadata`,
        `params`, `conditions`, `observations`, `theories`:
        >>> _init_result_list(metadata=VariableCollection(),
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


def resolve_state_params(state: SimpleCycleDataHistory) -> Dict:
    """
    Returns the `params` attribute of the input, with `cycle properties` resolved.

    Examples:

        >>> s = SimpleCycleDataHistory(theories=["the first theory", "the second theory"],
        ...     params={"experimentalist": {"source": "%theories[-1]%"}})
        >>> resolve_state_params(s)
        {'experimentalist': {'source': 'the second theory'}}

    """
    state_dependent_properties = _get_state_dependent_properties(state)
    resolved_params = _resolve_properties(state.params, state_dependent_properties)
    return resolved_params


def _get_state_dependent_properties(state: SimpleCycleDataHistory):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> state_dependent_properties = _get_state_dependent_properties(SimpleCycleDataHistory())

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

    n_ivs = len(state.metadata.independent_variables)
    n_dvs = len(state.metadata.dependent_variables)
    state_dependent_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: state.observations[-1][:, 0:n_ivs],
            "%observations.dvs[-1]%": lambda: state.observations[-1][:, n_ivs:],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + state.observations
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(state.observations)[:, n_ivs:],
            "%theories[-1]%": lambda: state.theories[-1],
            "%theories%": lambda: state.theories,
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

        >>> params_2 = {"key": "baz"}
        >>> _resolve_properties(params_2,cycle_properties_1)
        {'key': 'baz'}

    """
    params_ = copy.copy(params)
    for key, value in params_.items():
        if isinstance(value, dict):
            params_[key] = _resolve_properties(value, state_dependent_properties)
        elif isinstance(value, str) and (
            value in state_dependent_properties
        ):  # value is a key in the cycle_properties dictionary
            params_[key] = state_dependent_properties[value]
        else:
            pass  # no change needed

    return params_
