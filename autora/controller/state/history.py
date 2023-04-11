""" Classes for storing and passing a cycle's state as an immutable history. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.controller.protocol import (
    ResultKind,
    SupportsControllerStateHistory,
    SupportsDataKind,
)
from autora.controller.state.snapshot import Snapshot
from autora.variable import VariableCollection


class History(SupportsControllerStateHistory):
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
            >>> History()
            History([])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> History(metadata=VariableCollection()) # doctest: +ELLIPSIS
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            >>> History(params={"some": "params"})
            History([Result(data={'some': 'params'}, kind=ResultKind.PARAMETERS)])

            >>> History(conditions=["a condition"])
            History([Result(data='a condition', kind=ResultKind.EXPERIMENT)])

            >>> History(observations=["an observation"])
            History([Result(data='an observation', kind=ResultKind.OBSERVATION)])

            >>> from sklearn.linear_model import LinearRegression
            >>> History(theories=[LinearRegression()])
            History([Result(data=LinearRegression(), kind=ResultKind.THEORY)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `metadata`, `params`, `conditions`, `observations`, `theories`
            >>> History(theories=['t1', 't2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'}, metadata=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                                    Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                                    Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
                                    Result(data='c1', kind=ResultKind.EXPERIMENT),
                                    Result(data='c2', kind=ResultKind.EXPERIMENT),
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
            >>> h0 = History()
            >>> h0
            History([])

            We can update the metadata using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h1 = h0.update(metadata=VariableCollection())
            >>> h1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            ... the original object is unchanged:
            >>> h0
            History([])

            We can update the metadata again:
            >>> h2 = h1.update(metadata=VariableCollection(["some IV"]))
            >>> h2._by_kind  # doctest: +ELLIPSIS
            Snapshot(metadata=VariableCollection(independent_variables=['some IV'],...), ...)

            ... and we see that there is only ever one metadata object returned.

            Params is treated the same way as metadata:
            >>> hp = h0.update(params={'first': 'params'})
            >>> hp
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMETERS)])

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> hp = hp.update(params={'second': 'params'})
            >>> hp.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> hp  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMETERS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMETERS)])

            When we update the conditions, observations or theories, a new entry is added to the
            history:
            >>> h3 = h0.update(theories=["1st theory"])
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st theory', kind=ResultKind.THEORY)])

            ... so we can see the history of all the theories, for instance.
            >>> h3 = h3.update(theories=["2nd theory"])  # doctest: +NORMALIZE_WHITESPACE
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st theory', kind=ResultKind.THEORY),
                                    Result(data='2nd theory', kind=ResultKind.THEORY)])

            ... and the full history of theories is available using the `.theories` parameter:
            >>> h3.theories
            ['1st theory', '2nd theory']

            The same for the observations:
            >>> h4 = h0.update(observations=["1st observation"])
            >>> h4
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION)])

            >>> h4.update(observations=["2nd observation"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION),
                                    Result(data='2nd observation', kind=ResultKind.OBSERVATION)])


            The same for the conditions:
            >>> h5 = h0.update(conditions=["1st condition"])
            >>> h5
            History([Result(data='1st condition', kind=ResultKind.EXPERIMENT)])

            >>> h5.update(conditions=["2nd condition"])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st condition', kind=ResultKind.EXPERIMENT),
                                    Result(data='2nd condition', kind=ResultKind.EXPERIMENT)])

            You can also update with multiple conditions, observations and theories:
            >>> h0.update(conditions=['c1', 'c2'])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='c1', kind=ResultKind.EXPERIMENT),
                                    Result(data='c2', kind=ResultKind.EXPERIMENT)])

            >>> h0.update(theories=['t1', 't2'], metadata={'m': 1}) # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                                    Result(data='t1', kind=ResultKind.THEORY),
                                    Result(data='t2', kind=ResultKind.THEORY)])

            >>> h0.update(theories=['t1'], observations=['o1'], metadata={'m': 1}
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.THEORY)])

            We can also update with a complete history:
            >>> History().update(history=[Result(data={'m': 2}, kind=ResultKind.VARIABLES),
            ...                           Result(data='o1', kind=ResultKind.OBSERVATION),
            ...                           Result(data='t1', kind=ResultKind.THEORY)],
            ...                  conditions=['c1']
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 2}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.THEORY),
                     Result(data='c1', kind=ResultKind.EXPERIMENT)])

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

        return History(history=new_full_history)

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
            >>> h = History()

            ... and returns an emtpy metadata object
            >>> h.metadata
            VariableCollection(independent_variables=[], dependent_variables=[], covariates=[])

            We can update the metadata using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h = h.update(metadata=VariableCollection(independent_variables=['some IV']))
            >>> h.metadata  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some IV'], ...)

            We can update the metadata again:
            >>> h = h.update(metadata=VariableCollection(["some other IV"]))
            >>> h.metadata  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some other IV'], ...)

            ... and we see that there is only ever one metadata object returned."""
        return self._by_kind.metadata

    @property
    def params(self) -> Dict:
        """

        Returns:

        Examples:
            Params is treated the same way as metadata:
            >>> h = History()
            >>> h = h.update(params={'first': 'params'})
            >>> h.params
            {'first': 'params'}

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> h = h.update(params={'second': 'params'})
            >>> h.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> h  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMETERS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMETERS)])
        """
        return self._by_kind.params

    @property
    def conditions(self) -> List[ArrayLike]:
        """
        Returns:

        Examples:
            View the sequence of theories with one conditions:
            >>> h = History(conditions=[(1,2,3,)])
            >>> h.conditions
            [(1, 2, 3)]

            ... or more conditions:
            >>> h = h.update(conditions=[(4,5,6),(7,8,9)])  # doctest: +NORMALIZE_WHITESPACE
            >>> h.conditions
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        """
        return self._by_kind.conditions

    @property
    def observations(self) -> List[ArrayLike]:
        """

        Returns:

        Examples:
            The sequence of all observations is returned
            >>> h = History(observations=["1st observation"])
            >>> h.observations
            ['1st observation']

            >>> h = h.update(observations=["2nd observation"])
            >>> h.observations  # doctest: +ELLIPSIS
            ['1st observation', '2nd observation']

        """
        return self._by_kind.observations

    @property
    def theories(self) -> List[BaseEstimator]:
        """

        Returns:

        Examples:
            View the sequence of theories with one theory:
            >>> s = History(theories=["1st theory"])
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
            >>> h = History(theories=['t1', 't2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'}, metadata=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `metadata`, `params`, `conditions`, `observations`, `theories`

            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='from history', kind=ResultKind.VARIABLES),
             Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
             Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
             Result(data='c1', kind=ResultKind.EXPERIMENT),
             Result(data='c2', kind=ResultKind.EXPERIMENT),
             Result(data='o1', kind=ResultKind.OBSERVATION),
             Result(data='o2', kind=ResultKind.OBSERVATION),
             Result(data='t1', kind=ResultKind.THEORY),
             Result(data='t2', kind=ResultKind.THEORY)]

            If we add a new value, like the params object, the updated value is added to the
            end of the history:
            >>> h = h.update(params={'new': 'param'})
            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [..., Result(data={'new': 'param'}, kind=ResultKind.PARAMETERS)]

        """
        return self._history

    def filter_by(self, kind: Optional[Set[Union[str, ResultKind]]] = None) -> History:
        """
        Return a copy of the object with only data belonging to the specified kinds.

        Examples:
            >>> h = History(theories=['t1', 't2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'}, metadata=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            >>> h.filter_by(kind={"THEORY"})   # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='t1', kind=ResultKind.THEORY),
                                    Result(data='t2', kind=ResultKind.THEORY)])

            >>> h.filter_by(kind={ResultKind.OBSERVATION})  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION)])

            If we don't specify any filter criteria, we get the full history back:
            >>> h.filter_by()   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                     Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                     Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
                     Result(data='c1', kind=ResultKind.EXPERIMENT),
                     Result(data='c2', kind=ResultKind.EXPERIMENT),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='o2', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.THEORY),
                     Result(data='t2', kind=ResultKind.THEORY)])

        """
        if kind is None:
            return self
        else:
            kind_ = {ResultKind(s) for s in kind}
            filtered_history = _filter_history(self._history, kind_)
            new_object = History(history=filtered_history)
            return new_object


@dataclass(frozen=True)
class Result(SupportsDataKind):
    """
    Container class for data and metadata.

    Examples:
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
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)]

        >>> _init_result_list(params={"some": "params"})
        [Result(data={'some': 'params'}, kind=ResultKind.PARAMETERS)]

        >>> _init_result_list(conditions=["a condition"])
        [Result(data='a condition', kind=ResultKind.EXPERIMENT)]

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
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
         Result(data={'some': 'params'}, kind=ResultKind.PARAMETERS),
         Result(data='a condition', kind=ResultKind.EXPERIMENT),
         Result(data='an observation', kind=ResultKind.OBSERVATION),
         Result(data='another observation', kind=ResultKind.OBSERVATION),
         Result(data=LinearRegression(), kind=ResultKind.THEORY)]

    """
    data = []

    if metadata is not None:
        data.append(Result(metadata, ResultKind.VARIABLES))

    if params is not None:
        data.append(Result(params, ResultKind.PARAMETERS))

    for seq, kind in [
        (conditions, ResultKind.EXPERIMENT),
        (observations, ResultKind.OBSERVATION),
        (theories, ResultKind.THEORY),
    ]:
        if seq is not None:
            for i in seq:
                data.append(Result(i, kind=kind))

    return data


def _history_to_kind(history: Sequence[Result]) -> Snapshot:
    """
    Convert a sequence of results into a Snapshot instance:

    Examples:
        History might be empty
        >>> history_ = []
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(metadata=VariableCollection(...), params={},
                        conditions=[], observations=[], theories=[])

        ... or with values for any or all of the parameters:
        >>> history_ = _init_result_list(params={"some": "params"})
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, ...)

        >>> history_ += _init_result_list(conditions=["a condition"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, conditions=['a condition'], ...)

        >>> _history_to_kind(history_).params
        {'some': 'params'}

        >>> history_ += _init_result_list(observations=["an observation"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, conditions=['a condition'],
                        observations=['an observation'], ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Result(LinearRegression(), kind=ResultKind.THEORY)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., theories=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> metadata = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Result(metadata, kind=ResultKind.VARIABLES)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(metadata=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Result({'some': 'params'}, kind=ResultKind.PARAMETERS)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, ...)

    """
    namespace = Snapshot(
        metadata=_get_last_data_with_default(
            history, kind={ResultKind.VARIABLES}, default=VariableCollection()
        ),
        params=_get_last_data_with_default(
            history, kind={ResultKind.PARAMETERS}, default={}
        ),
        observations=_list_data(
            _filter_history(history, kind={ResultKind.OBSERVATION})
        ),
        theories=_list_data(_filter_history(history, kind={ResultKind.THEORY})),
        conditions=_list_data(_filter_history(history, kind={ResultKind.EXPERIMENT})),
    )
    return namespace


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


def _filter_history(data: Iterable[SupportsDataKind], kind: Set[ResultKind]):
    return filter(lambda r: r.kind in kind, data)


def _get_last(data: Sequence[SupportsDataKind], kind: Set[ResultKind]):
    results_new_to_old = reversed(data)
    last_of_kind = next(_filter_history(results_new_to_old, kind=kind))
    return last_of_kind


def _get_last_data_with_default(data: Sequence[SupportsDataKind], kind, default):
    try:
        result = _get_last(data, kind).data
    except StopIteration:
        result = default
    return result
