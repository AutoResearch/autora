""" Classes for storing and passing a cycle's state as an immutable history. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.controller.protocol import (
    RecordKind,
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
        history: Optional[Sequence[Record]] = None,
    ):
        """

        Args:
            variables: a single datum to be marked as "variables"
            parameters: a single datum to be marked as "parameters"
            experiments: an iterable of data, each to be marked as "experiments"
            observations: an iterable of data, each to be marked as "observations"
            models: an iterable of data, each to be marked as "models"
            history: an iterable of Record objects to be used as the initial history.

        Examples:
            Empty input leads to an empty state:
            >>> History()
            History([])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> History(variables=VariableCollection()) # doctest: +ELLIPSIS
            History([Record(data=VariableCollection(...), kind=RecordKind.VARIABLES)])

            >>> History(parameters={"some": "parameters"})
            History([Record(data={'some': 'parameters'}, kind=RecordKind.PARAMETERS)])

            >>> History(experiments=["a experiment"])
            History([Record(data='a experiment', kind=RecordKind.EXPERIMENT)])

            >>> History(observations=["an observation"])
            History([Record(data='an observation', kind=RecordKind.OBSERVATION)])

            >>> from sklearn.linear_model import LinearRegression
            >>> History(models=[LinearRegression()])
            History([Record(data=LinearRegression(), kind=RecordKind.MODEL)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `parameters`, `experiments`, `observations`, `models`
            >>> History(models=['m1', 'm2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Record("from history", RecordKind.VARIABLES)]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Record(data='from history', kind=RecordKind.VARIABLES),
                     Record(data=VariableCollection(...), kind=RecordKind.VARIABLES),
                     Record(data={'a': 'param'}, kind=RecordKind.PARAMETERS),
                     Record(data='e1', kind=RecordKind.EXPERIMENT),
                     Record(data='e2', kind=RecordKind.EXPERIMENT),
                     Record(data='o1', kind=RecordKind.OBSERVATION),
                     Record(data='o2', kind=RecordKind.OBSERVATION),
                     Record(data='m1', kind=RecordKind.MODEL),
                     Record(data='m2', kind=RecordKind.MODEL)])
        """
        self._history: List

        if history is not None:
            self._history = list(history)
        else:
            self._history = []

        self._history += _init_record_list(
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
            History([Record(data=VariableCollection(...), kind=RecordKind.VARIABLES)])

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
            History([Record(data={'first': 'parameters'}, kind=RecordKind.PARAMETERS)])

            ... where only the most recent "parameters" object is returned from the
            `.parameters` property.
            >>> hp = hp.update(parameters={'second': 'parameters'})
            >>> hp.parameters
            {'second': 'parameters'}

            ... however, the full history of the parameters objects remains available, if needed:
            >>> hp  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data={'first': 'parameters'}, kind=RecordKind.PARAMETERS),
                     Record(data={'second': 'parameters'}, kind=RecordKind.PARAMETERS)])

            When we update the experiments, observations or models, a new entry is added to the
            history:
            >>> h3 = h0.update(models=["1st model"])
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='1st model', kind=RecordKind.MODEL)])

            ... so we can see the history of all the models, for instance.
            >>> h3 = h3.update(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='1st model', kind=RecordKind.MODEL),
                                    Record(data='2nd model', kind=RecordKind.MODEL)])

            ... and the full history of models is available using the `.models` parameter:
            >>> h3.models
            ['1st model', '2nd model']

            The same for the observations:
            >>> h4 = h0.update(observations=["1st observation"])
            >>> h4
            History([Record(data='1st observation', kind=RecordKind.OBSERVATION)])

            >>> h4.update(observations=["2nd observation"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Record(data='1st observation', kind=RecordKind.OBSERVATION),
                                    Record(data='2nd observation', kind=RecordKind.OBSERVATION)])


            The same for the experiments:
            >>> h5 = h0.update(experiments=["1st experiment"])
            >>> h5
            History([Record(data='1st experiment', kind=RecordKind.EXPERIMENT)])

            >>> h5.update(experiments=["2nd experiment"])  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='1st experiment', kind=RecordKind.EXPERIMENT),
                                    Record(data='2nd experiment', kind=RecordKind.EXPERIMENT)])

            You can also update with multiple experiments, observations and models:
            >>> h0.update(experiments=['e1', 'e2'])  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='e1', kind=RecordKind.EXPERIMENT),
                                    Record(data='e2', kind=RecordKind.EXPERIMENT)])

            >>> h0.update(models=['m1', 'm2'], variables={'m': 1}
            ... ) # doctest: +NORMALIZE_WHITESPACE
            History([Record(data={'m': 1}, kind=RecordKind.VARIABLES),
                                    Record(data='m1', kind=RecordKind.MODEL),
                                    Record(data='m2', kind=RecordKind.MODEL)])

            >>> h0.update(models=['m1'], observations=['o1'], variables={'m': 1}
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data={'m': 1}, kind=RecordKind.VARIABLES),
                     Record(data='o1', kind=RecordKind.OBSERVATION),
                     Record(data='m1', kind=RecordKind.MODEL)])

            We can also update with a complete history:
            >>> History().update(history=[Record(data={'m': 2}, kind=RecordKind.VARIABLES),
            ...                           Record(data='o1', kind=RecordKind.OBSERVATION),
            ...                           Record(data='m1', kind=RecordKind.MODEL)],
            ...                  experiments=['e1']
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data={'m': 2}, kind=RecordKind.VARIABLES),
                     Record(data='o1', kind=RecordKind.OBSERVATION),
                     Record(data='m1', kind=RecordKind.MODEL),
                     Record(data='e1', kind=RecordKind.EXPERIMENT)])

        """

        if history is not None:
            history_extension = history
        else:
            history_extension = []

        history_extension += _init_record_list(
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
            History([Record(data={'first': 'parameters'}, kind=RecordKind.PARAMETERS),
                     Record(data={'second': 'parameters'}, kind=RecordKind.PARAMETERS)])
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
    def history(self) -> List[Record]:
        """

        Examples:
            We initialze some history:
            >>> h = History(models=['m1', 'm2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Record("from history", RecordKind.VARIABLES)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `parameters`, `experiments`, `observations`, `models`

            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Record(data='from history', kind=RecordKind.VARIABLES),
             Record(data=VariableCollection(...), kind=RecordKind.VARIABLES),
             Record(data={'a': 'param'}, kind=RecordKind.PARAMETERS),
             Record(data='e1', kind=RecordKind.EXPERIMENT),
             Record(data='e2', kind=RecordKind.EXPERIMENT),
             Record(data='o1', kind=RecordKind.OBSERVATION),
             Record(data='o2', kind=RecordKind.OBSERVATION),
             Record(data='m1', kind=RecordKind.MODEL),
             Record(data='m2', kind=RecordKind.MODEL)]

            If we add a new value, like the parameters object, the updated value is added to the
            end of the history:
            >>> h = h.update(parameters={'new': 'param'})
            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [..., Record(data={'new': 'param'}, kind=RecordKind.PARAMETERS)]

        """
        return self._history

    def filter_by(self, kind: Optional[Set[Union[str, RecordKind]]] = None) -> History:
        """
        Return a copy of the object with only data belonging to the specified kinds.

        Examples:
            >>> h = History(models=['m1', 'm2'], experiments=['e1', 'e2'],
            ...     observations=['o1', 'o2'], parameters={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Record("from history", RecordKind.VARIABLES)])

            >>> h.filter_by(kind={"MODEL"})   # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='m1', kind=RecordKind.MODEL),
                                    Record(data='m2', kind=RecordKind.MODEL)])

            >>> h.filter_by(kind={RecordKind.OBSERVATION})  # doctest: +NORMALIZE_WHITESPACE
            History([Record(data='o1', kind=RecordKind.OBSERVATION),
                                    Record(data='o2', kind=RecordKind.OBSERVATION)])

            If we don't specify any filter criteria, we get the full history back:
            >>> h.filter_by()   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            History([Record(data='from history', kind=RecordKind.VARIABLES),
                     Record(data=VariableCollection(...), kind=RecordKind.VARIABLES),
                     Record(data={'a': 'param'}, kind=RecordKind.PARAMETERS),
                     Record(data='e1', kind=RecordKind.EXPERIMENT),
                     Record(data='e2', kind=RecordKind.EXPERIMENT),
                     Record(data='o1', kind=RecordKind.OBSERVATION),
                     Record(data='o2', kind=RecordKind.OBSERVATION),
                     Record(data='m1', kind=RecordKind.MODEL),
                     Record(data='m2', kind=RecordKind.MODEL)])

        """
        if kind is None:
            return self
        else:
            kind_ = {RecordKind(s) for s in kind}
            filtered_history = _filter_history(self._history, kind_)
            new_object = History(history=filtered_history)
            return new_object


@dataclass(frozen=True)
class Record(SupportsDataKind):
    """
    Container class for data and variables.

    Examples:
        >>> Record()
        Record(data=None, kind=None)

        >>> Record("a")
        Record(data='a', kind=None)

        >>> Record(None, "MODEL")
        Record(data=None, kind=RecordKind.MODEL)

        >>> Record(data="b")
        Record(data='b', kind=None)

        >>> Record("c", "OBSERVATION")
        Record(data='c', kind=RecordKind.OBSERVATION)
    """

    data: Optional[Any] = None
    kind: Optional[RecordKind] = None

    def __post_init__(self):
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", RecordKind(self.kind))


def _init_record_list(
    variables: Optional[VariableCollection] = None,
    parameters: Optional[Dict] = None,
    experiments: Optional[Iterable[ArrayLike]] = None,
    observations: Optional[Iterable[ArrayLike]] = None,
    models: Optional[Iterable[BaseEstimator]] = None,
) -> List[Record]:
    """
    Initialize a list of Record objects

    Returns:

    Args:
        variables: a single datum to be marked as "variables"
        parameters: a single datum to be marked as "parameters"
        experiments: an iterable of data, each to be marked as "experiments"
        observations: an iterable of data, each to be marked as "observations"
        models: an iterable of data, each to be marked as "models"

    Examples:
        Empty input leads to an empty state:
        >>> _init_record_list()
        []

        ... or with values for any or all of the parameters:
        >>> from autora.variable import VariableCollection
        >>> _init_record_list(variables=VariableCollection()) # doctest: +ELLIPSIS
        [Record(data=VariableCollection(...), kind=RecordKind.VARIABLES)]

        >>> _init_record_list(parameters={"some": "parameters"})
        [Record(data={'some': 'parameters'}, kind=RecordKind.PARAMETERS)]

        >>> _init_record_list(experiments=["a experiment"])
        [Record(data='a experiment', kind=RecordKind.EXPERIMENT)]

        >>> _init_record_list(observations=["an observation"])
        [Record(data='an observation', kind=RecordKind.OBSERVATION)]

        >>> from sklearn.linear_model import LinearRegression
        >>> _init_record_list(models=[LinearRegression()])
        [Record(data=LinearRegression(), kind=RecordKind.MODEL)]

        The input arguments are added to the data in the order `variables`,
        `parameters`, `experiments`, `observations`, `models`:
        >>> _init_record_list(variables=VariableCollection(),
        ...                  parameters={"some": "parameters"},
        ...                  experiments=["a experiment"],
        ...                  observations=["an observation", "another observation"],
        ...                  models=[LinearRegression()],
        ... ) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [Record(data=VariableCollection(...), kind=RecordKind.VARIABLES),
         Record(data={'some': 'parameters'}, kind=RecordKind.PARAMETERS),
         Record(data='a experiment', kind=RecordKind.EXPERIMENT),
         Record(data='an observation', kind=RecordKind.OBSERVATION),
         Record(data='another observation', kind=RecordKind.OBSERVATION),
         Record(data=LinearRegression(), kind=RecordKind.MODEL)]

    """
    data = []

    if variables is not None:
        data.append(Record(variables, RecordKind.VARIABLES))

    if parameters is not None:
        data.append(Record(parameters, RecordKind.PARAMETERS))

    for seq, kind in [
        (experiments, RecordKind.EXPERIMENT),
        (observations, RecordKind.OBSERVATION),
        (models, RecordKind.MODEL),
    ]:
        if seq is not None:
            for i in seq:
                data.append(Record(i, kind=kind))

    return data


def _history_to_kind(history: Sequence[Record]) -> Snapshot:
    """
    Convert a sequence of results into a Snapshot instance:

    Examples:
        History might be empty
        >>> history_ = []
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(variables=VariableCollection(...), parameters={},
                        experiments=[], observations=[], models=[])

        ... or with values for any or all of the parameters:
        >>> history_ = _init_record_list(parameters={"some": "parameters"})
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, ...)

        >>> history_ += _init_record_list(experiments=["a experiment"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, experiments=['a experiment'], ...)

        >>> _history_to_kind(history_).parameters
        {'some': 'parameters'}

        >>> history_ += _init_record_list(observations=["an observation"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, experiments=['a experiment'],
                        observations=['an observation'], ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Record(LinearRegression(), kind=RecordKind.MODEL)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., models=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> variables = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Record(variables, kind=RecordKind.VARIABLES)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(variables=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Record({'some': 'parameters'}, kind=RecordKind.PARAMETERS)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., parameters={'some': 'parameters'}, ...)

    """
    namespace = Snapshot(
        variables=_get_last_data_with_default(
            history, kind={RecordKind.VARIABLES}, default=VariableCollection()
        ),
        parameters=_get_last_data_with_default(
            history, kind={RecordKind.PARAMETERS}, default={}
        ),
        observations=_list_data(
            _filter_history(history, kind={RecordKind.OBSERVATION})
        ),
        models=_list_data(_filter_history(history, kind={RecordKind.MODEL})),
        experiments=_list_data(_filter_history(history, kind={RecordKind.EXPERIMENT})),
    )
    return namespace


def _list_data(data: Sequence[SupportsDataKind]):
    """
    Extract the `.data` attribute of each item in a sequence, and return as a list.

    Examples:
        >>> _list_data([])
        []

        >>> _list_data([Record("a"), Record("b")])
        ['a', 'b']
    """
    return list(r.data for r in data)


def _filter_history(data: Iterable[SupportsDataKind], kind: Set[RecordKind]):
    return filter(lambda r: r.kind in kind, data)


def _get_last(data: Sequence[SupportsDataKind], kind: Set[RecordKind]):
    results_new_to_old = reversed(data)
    last_of_kind = next(_filter_history(results_new_to_old, kind=kind))
    return last_of_kind


def _get_last_data_with_default(data: Sequence[SupportsDataKind], kind, default):
    try:
        result = _get_last(data, kind).data
    except StopIteration:
        result = default
    return result
