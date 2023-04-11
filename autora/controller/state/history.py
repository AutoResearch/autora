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
        variables: Optional[VariableCollection] = None,
        parameters: Optional[Dict] = None,
        experiments: Optional[List[ArrayLike]] = None,
        observations: Optional[List[ArrayLike]] = None,
        models: Optional[List[BaseEstimator]] = None,
        history: Optional[Sequence[Result]] = None,
    ):
        """

        Args:
            variables: a single datum to be marked as "variables"
            parameters: a single datum to be marked as "parameters"
            experiments: an iterable of data, each to be marked as "experiments"
            observations: an iterable of data, each to be marked as "observations"
            models: an iterable of data, each to be marked as "models"
            history: an iterable of Result objects to be used as the initial history.

        Examples:
            Empty input leads to an empty state:
            >>> History()
            History([])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> History(variables=VariableCollection()) # doctest: +ELLIPSIS
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            >>> History(parameters={"some": "parameters"})
            History([Result(data={'some': 'parameters'}, kind=ResultKind.PARAMETERS)])

            >>> History(experiments=["a experiment"])
            History([Result(data='a experiment', kind=ResultKind.EXPERIMENT)])

            >>> History(observations=["an observation"])
            History([Result(data='an observation', kind=ResultKind.OBSERVATION)])

            >>> from sklearn.linear_model import LinearRegression
            >>> History(models=[LinearRegression()])
            History([Result(data=LinearRegression(), kind=ResultKind.MODEL)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `parameters`, `experiments`, `observations`, `models`
            >>> History(models=['t1', 't2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                                    Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                                    Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
                                    Result(data='e1', kind=ResultKind.EXPERIMENT),
                                    Result(data='e2', kind=ResultKind.EXPERIMENT),
                                    Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION),
                                    Result(data='t1', kind=ResultKind.MODEL),
                                    Result(data='t2', kind=ResultKind.MODEL)])
        """
        self._history: List

        if history is not None:
            self._history = list(history)
        else:
            self._history = []

        self._history += _init_result_list(
            variables=variables,
            parameters=parameters,
            experiments=experiments,
            observations=observations,
            models=models,
        )

    def update(
        self,
        variables=None,
        parameters=None,
        experiments=None,
        observations=None,
        models=None,
        history=None,
    ):
        """
        Create a new object with updated values.

        Examples:
            The initial object is empty:
            >>> h0 = History()
            >>> h0
            History([])

            We can update the variables using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h1 = h0.update(variables=VariableCollection())
            >>> h1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            ... the original object is unchanged:
            >>> h0
            History([])

            We can update the variables again:
            >>> h2 = h1.update(variables=VariableCollection(["some IV"]))
            >>> h2._by_kind  # doctest: +ELLIPSIS
            Snapshot(variables=VariableCollection(independent_variables=['some IV'],...), ...)

            ... and we see that there is only ever one variables object returned.

            Params is treated the same way as variables:
            >>> hp = h0.update(parameters={'first': 'parameters'})
            >>> hp
            History([Result(data={'first': 'parameters'}, kind=ResultKind.PARAMETERS)])

            ... where only the most recent "parameters" object is returned from the
            `.parameters` property.
            >>> hp = hp.update(parameters={'second': 'parameters'})
            >>> hp.parameters
            {'second': 'parameters'}

            ... however, the full history of the parameters objects remains available, if needed:
            >>> hp  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'parameters'}, kind=ResultKind.PARAMETERS),
                     Result(data={'second': 'parameters'}, kind=ResultKind.PARAMETERS)])

            When we update the experiments, observations or models, a new entry is added to the
            history:
            >>> h3 = h0.update(models=["1st model"])
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st model', kind=ResultKind.MODEL)])

            ... so we can see the history of all the models, for instance.
            >>> h3 = h3.update(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st model', kind=ResultKind.MODEL),
                                    Result(data='2nd model', kind=ResultKind.MODEL)])

            ... and the full history of models is available using the `.models` parameter:
            >>> h3.models
            ['1st model', '2nd model']

            The same for the observations:
            >>> h4 = h0.update(observations=["1st observation"])
            >>> h4
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION)])

            >>> h4.update(observations=["2nd observation"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION),
                                    Result(data='2nd observation', kind=ResultKind.OBSERVATION)])


            The same for the experiments:
            >>> h5 = h0.update(experiments=["1st experiment"])
            >>> h5
            History([Result(data='1st experiment', kind=ResultKind.EXPERIMENT)])

            >>> h5.update(experiments=["2nd experiment"])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st experiment', kind=ResultKind.EXPERIMENT),
                                    Result(data='2nd experiment', kind=ResultKind.EXPERIMENT)])

            You can also update with multiple experiments, observations and models:
            >>> h0.update(experiments=['e1', 'e2'])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='e1', kind=ResultKind.EXPERIMENT),
                                    Result(data='e2', kind=ResultKind.EXPERIMENT)])

            >>> h0.update(models=['t1', 't2'], variables={'m': 1}
            ... ) # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                                    Result(data='t1', kind=ResultKind.MODEL),
                                    Result(data='t2', kind=ResultKind.MODEL)])

            >>> h0.update(models=['t1'], observations=['o1'], variables={'m': 1}
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.MODEL)])

            We can also update with a complete history:
            >>> History().update(history=[Result(data={'m': 2}, kind=ResultKind.VARIABLES),
            ...                           Result(data='o1', kind=ResultKind.OBSERVATION),
            ...                           Result(data='t1', kind=ResultKind.MODEL)],
            ...                  experiments=['e1']
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 2}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.MODEL),
                     Result(data='e1', kind=ResultKind.EXPERIMENT)])

        """

        if history is not None:
            history_extension = history
        else:
            history_extension = []

        history_extension += _init_result_list(
            variables=variables,
            parameters=parameters,
            experiments=experiments,
            observations=observations,
            models=models,
        )
        new_full_history = self._history + history_extension

        return History(history=new_full_history)

    def __repr__(self):
        return f"{type(self).__name__}({self.history})"

    @property
    def _by_kind(self):
        return _history_to_kind(self._history)

    @property
    def variables(self) -> VariableCollection:
        """

        Examples:
            The initial object is empty:
            >>> h = History()

            ... and returns an emtpy variables object
            >>> h.variables
            VariableCollection(independent_variables=[], dependent_variables=[], covariates=[])

            We can update the variables using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h = h.update(variables=VariableCollection(independent_variables=['some IV']))
            >>> h.variables  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some IV'], ...)

            We can update the variables again:
            >>> h = h.update(variables=VariableCollection(["some other IV"]))
            >>> h.variables  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some other IV'], ...)

            ... and we see that there is only ever one variables object returned."""
        return self._by_kind.variables

    @property
    def parameters(self) -> Dict:
        """

        Returns:

        Examples:
            Params is treated the same way as variables:
            >>> h = History()
            >>> h = h.update(parameters={'first': 'parameters'})
            >>> h.parameters
            {'first': 'parameters'}

            ... where only the most recent "parameters" object is returned
            from the `.parameters` property.
            >>> h = h.update(parameters={'second': 'parameters'})
            >>> h.parameters
            {'second': 'parameters'}

            ... however, the full history of the parameters objects remains available, if needed:
            >>> h  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'parameters'}, kind=ResultKind.PARAMETERS),
                     Result(data={'second': 'parameters'}, kind=ResultKind.PARAMETERS)])
        """
        return self._by_kind.parameters

    @property
    def experiments(self) -> List[ArrayLike]:
        """
        Returns:

        Examples:
            View the sequence of models with one experiments:
            >>> h = History(experiments=[(1,2,3,)])
            >>> h.experiments
            [(1, 2, 3)]

            ... or more experiments:
            >>> h = h.update(experiments=[(4,5,6),(7,8,9)])  # doctest: +NORMALIZE_WHITESPACE
            >>> h.experiments
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        """
        return self._by_kind.experiments

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
    def models(self) -> List[BaseEstimator]:
        """

        Returns:

        Examples:
            View the sequence of models with one model:
            >>> s = History(models=["1st model"])
            >>> s.models  # doctest: +NORMALIZE_WHITESPACE
            ['1st model']

            ... or more models:
            >>> s = s.update(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
            >>> s.models
            ['1st model', '2nd model']

        """
        return self._by_kind.models

    @property
    def history(self) -> List[Result]:
        """

        Examples:
            We initialze some history:
            >>> h = History(models=['t1', 't2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `parameters`, `experiments`, `observations`, `models`

            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='from history', kind=ResultKind.VARIABLES),
             Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
             Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
             Result(data='e1', kind=ResultKind.EXPERIMENT),
             Result(data='e2', kind=ResultKind.EXPERIMENT),
             Result(data='o1', kind=ResultKind.OBSERVATION),
             Result(data='o2', kind=ResultKind.OBSERVATION),
             Result(data='t1', kind=ResultKind.MODEL),
             Result(data='t2', kind=ResultKind.MODEL)]

            If we add a new value, like the parameters object, the updated value is added to the
            end of the history:
            >>> h = h.update(parameters={'new': 'param'})
            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [..., Result(data={'new': 'param'}, kind=ResultKind.PARAMETERS)]

        """
        return self._history

    def filter_by(self, kind: Optional[Set[Union[str, ResultKind]]] = None) -> History:
        """
        Return a copy of the object with only data belonging to the specified kinds.

        Examples:
            >>> h = History(models=['t1', 't2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            >>> h.filter_by(kind={"MODEL"})   # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='t1', kind=ResultKind.MODEL),
                                    Result(data='t2', kind=ResultKind.MODEL)])

            >>> h.filter_by(kind={ResultKind.OBSERVATION})  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION)])

            If we don't specify any filter criteria, we get the full history back:
            >>> h.filter_by()   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                     Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                     Result(data={'a': 'param'}, kind=ResultKind.PARAMETERS),
                     Result(data='e1', kind=ResultKind.EXPERIMENT),
                     Result(data='e2', kind=ResultKind.EXPERIMENT),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='o2', kind=ResultKind.OBSERVATION),
                     Result(data='t1', kind=ResultKind.MODEL),
                     Result(data='t2', kind=ResultKind.MODEL)])

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
    Container class for data and variables.

    Examples:
        >>> Result()
        Result(data=None, kind=None)

        >>> Result("a")
        Result(data='a', kind=None)

        >>> Result(None, "MODEL")
        Result(data=None, kind=ResultKind.MODEL)

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
    variables: Optional[VariableCollection] = None,
    parameters: Optional[Dict] = None,
    experiments: Optional[Iterable[ArrayLike]] = None,
    observations: Optional[Iterable[ArrayLike]] = None,
    models: Optional[Iterable[BaseEstimator]] = None,
) -> List[Result]:
    """
    Initialize a list of Result objects

    Returns:

    Args:
        variables: a single datum to be marked as "variables"
        parameters: a single datum to be marked as "parameters"
        experiments: an iterable of data, each to be marked as "experiments"
        observations: an iterable of data, each to be marked as "observations"
        models: an iterable of data, each to be marked as "models"

    Examples:
        Empty input leads to an empty state:
        >>> _init_result_list()
        []

        ... or with values for any or all of the parameters:
        >>> from autora.variable import VariableCollection
        >>> _init_result_list(variables=VariableCollection()) # doctest: +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)]

        >>> _init_result_list(parameters={"some": "parameters"})
        [Result(data={'some': 'parameters'}, kind=ResultKind.PARAMETERS)]

        >>> _init_result_list(experiments=["a experiment"])
        [Result(data='a experiment', kind=ResultKind.EXPERIMENT)]

        >>> _init_result_list(observations=["an observation"])
        [Result(data='an observation', kind=ResultKind.OBSERVATION)]

        >>> from sklearn.linear_model import LinearRegression
        >>> _init_result_list(models=[LinearRegression()])
        [Result(data=LinearRegression(), kind=ResultKind.MODEL)]

        The input arguments are added to the data in the order `variables`,
        `parameters`, `experiments`, `observations`, `models`:
        >>> _init_result_list(variables=VariableCollection(),
        ...                  parameters={"some": "parameters"},
        ...                  experiments=["a experiment"],
        ...                  observations=["an observation", "another observation"],
        ...                  models=[LinearRegression()],
        ... ) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
         Result(data={'some': 'parameters'}, kind=ResultKind.PARAMETERS),
         Result(data='a experiment', kind=ResultKind.EXPERIMENT),
         Result(data='an observation', kind=ResultKind.OBSERVATION),
         Result(data='another observation', kind=ResultKind.OBSERVATION),
         Result(data=LinearRegression(), kind=ResultKind.MODEL)]

    """
    data = []

    if variables is not None:
        data.append(Result(variables, ResultKind.VARIABLES))

    if parameters is not None:
        data.append(Result(parameters, ResultKind.PARAMETERS))

    for seq, kind in [
        (experiments, ResultKind.EXPERIMENT),
        (observations, ResultKind.OBSERVATION),
        (models, ResultKind.MODEL),
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
        Snapshot(variables=VariableCollection(...), parameters={},
                        experiments=[], observations=[], models=[])

        ... or with values for any or all of the parameters:
        >>> history_ = _init_result_list(parameters={"some": "parameters"})
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, ...)

        >>> history_ += _init_result_list(experiments=["a experiment"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, experiments=['a experiment'], ...)

        >>> _history_to_kind(history_).parameters
        {'some': 'parameters'}

        >>> history_ += _init_result_list(observations=["an observation"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, experiments=['a experiment'],
                        observations=['an observation'], ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Result(LinearRegression(), kind=ResultKind.MODEL)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., models=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> variables = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Result(variables, kind=ResultKind.VARIABLES)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(variables=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Result({'some': 'parameters'}, kind=ResultKind.PARAMETERS)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, ...)

    """
    namespace = Snapshot(
        variables=_get_last_data_with_default(
            history, kind={ResultKind.VARIABLES}, default=VariableCollection()
        ),
        parameters=_get_last_data_with_default(
            history, kind={ResultKind.PARAMETERS}, default={}
        ),
        observations=_list_data(
            _filter_history(history, kind={ResultKind.OBSERVATION})
        ),
        models=_list_data(_filter_history(history, kind={ResultKind.MODEL})),
        experiments=_list_data(_filter_history(history, kind={ResultKind.EXPERIMENT})),
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
