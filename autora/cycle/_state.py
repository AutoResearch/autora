from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from numpy._typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


@dataclass
class CycleState:
    """Container class for the state of an AER cycle."""

    data: List[Result]

    def __init__(
        self,
        metadata: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        conditions: Optional[Sequence[ArrayLike]] = None,
        observations: Optional[Sequence[ArrayLike]] = None,
        theories: Optional[Sequence[BaseEstimator]] = None,
        data: Optional[Sequence[Result]] = None,
    ):
        """

        Args:
            metadata: a single datum to be marked as "metadata"
            params: a single datum to be marked as "params"
            conditions: a sequence of data, each to be marked as "conditions"
            observations: a sequence of data, each to be marked as "observations"
            theories: a sequence of data, each to be marked as "theories"
            data: a sequence of `Result` objects

        Examples:
            CycleState can be initialized in an empty state:
            >>> CycleState()
            CycleState(data=[])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> CycleState(metadata=VariableCollection())
            CycleState(data=[Result(data=VariableCollection(...), kind=ResultKind.METADATA)])

            >>> CycleState(params={"some": "params"})
            CycleState(data=[Result(data={'some': 'params'}, kind=ResultKind.PARAMS)])

            >>> CycleState(conditions=["a condition"])
            CycleState(data=[Result(data='a condition', kind=ResultKind.CONDITION)])

            >>> CycleState(observations=["an observation"])
            CycleState(data=[Result(data='an observation', kind=ResultKind.OBSERVATION)])

            >>> CycleState(theories=["a theory"])
            CycleState(data=[Result(data='a theory', kind=ResultKind.THEORY)])

            The CycleState can also be initialized with a sequence of `Result` objects:
            >>> data_generator = (Result(i, ResultKind.CONDITION) for i in range(5))
            >>> CycleState(data=data_generator) # doctest: +NORMALIZE_WHITESPACE
            CycleState(data=[Result(data=0, kind=ResultKind.CONDITION),
                             Result(data=1, kind=ResultKind.CONDITION),
                             Result(data=2, kind=ResultKind.CONDITION),
                             Result(data=3, kind=ResultKind.CONDITION),
                             Result(data=4, kind=ResultKind.CONDITION)])

            The input arguments are added to the data in the order `data`, `metadata`,
            `params`, `conditions`, `observations`, `theories`:
            >>> CycleState(metadata=VariableCollection(),
            ...            params={"some": "params"},
            ...            conditions=["a condition"],
            ...            observations=["an observation", "another observation"],
            ...            theories=["a theory"],
            ...            data=(Result(i, ResultKind.CONDITION) for i in range(2))
            ... ) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            CycleState(data=[Result(data=0, kind=ResultKind.CONDITION),
                             Result(data=1, kind=ResultKind.CONDITION),
                             Result(data=VariableCollection(...), kind=ResultKind.METADATA),
                             Result(data={'some': 'params'}, kind=ResultKind.PARAMS),
                             Result(data='a condition', kind=ResultKind.CONDITION),
                             Result(data='an observation', kind=ResultKind.OBSERVATION),
                             Result(data='another observation', kind=ResultKind.OBSERVATION),
                             Result(data='a theory', kind=ResultKind.THEORY)])

        """
        if data is not None:
            self.data = list(data)
        else:
            self.data = []

        if metadata is not None:
            self.metadata = metadata

        if params is not None:
            self.params = params

        for seq, kind in [
            (conditions, ResultKind.CONDITION),
            (observations, ResultKind.OBSERVATION),
            (theories, ResultKind.THEORY),
        ]:
            if seq is not None:
                for i in seq:
                    self.update(i, kind=kind)

    def update(self, value, kind: Union[ResultKind, str]):
        """
        Add a new value.

        Examples:
            First, create an empty state:
            >>> state = CycleState()

            Now add data and any metadata required. The `kind` can be specified as a string:
            >>> state.update("first metadata", kind="METADATA")
            >>> state.metadata
            'first metadata'

            ... or by using the ResultKind enum. The effect is the same.
            >>> state.update("second metadata", kind=ResultKind.METADATA)
            >>> state.metadata
            'second metadata'

            >>> state.update("first theory", kind="THEORY")
            >>> state.theories
            ['first theory']

            >>> state.update("first condition", kind="CONDITION")
            >>> state.conditions
            ['first condition']

            >>> state.results  # doctest: +NORMALIZE_WHITESPACE
            [Result(data='first theory', kind=ResultKind.THEORY),
             Result(data='first condition', kind=ResultKind.CONDITION)]

            >>> state.data  # doctest: +NORMALIZE_WHITESPACE
            [Result(data='first metadata',  kind=ResultKind.METADATA),
             Result(data='second metadata', kind=ResultKind.METADATA),
             Result(data='first theory',    kind=ResultKind.THEORY),
             Result(data='first condition', kind=ResultKind.CONDITION)]

        """
        self.data.append(Result(value, ResultKind(kind)))

    @property
    def metadata(self) -> VariableCollection:
        """
        Access the newest metadata.

        Examples:
             >>> state = CycleState()

             If we ask for a value which isn't set, then we get a default empty VariableCollection:
             >>> state.metadata
             VariableCollection(independent_variables=[], dependent_variables=[], covariates=[])

             We can add a new metadata object by using the setter:
             >>> from autora.variable import IV
             >>> state.metadata = VariableCollection([IV(name="IV1")])
             >>> state.metadata  # doctest: +ELLIPSIS
             VariableCollection(independent_variables=[IV(name='IV1', ...)], ...)

             ... or by calling "update" directly:
             >>> state.update("new metadata", kind=ResultKind.METADATA)

             That adds the metadata object to the .data attribute:
             >>> state.data  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
             [Result(data=VariableCollection(independent_variables=[IV(name='IV1',...),
                     kind=ResultKind.METADATA),
              Result(data='new metadata', kind=ResultKind.METADATA)]

              If we access the metadata property, we just get the newest one back:
              >>> state.metadata
              'new metadata'

        """
        try:
            m = self._get_last(kind={ResultKind.METADATA}).data
        except StopIteration:
            m = VariableCollection()
        return m

    @metadata.setter
    def metadata(self, value):
        self.update(value, kind=ResultKind.METADATA)

    @property
    def params(self) -> Dict:
        """
        Access the newest params.

        Examples:
             >>> state = CycleState()

             If we ask for a value which isn't set, then we get a default empty dictionary:
             >>> state.params
             {}

             We can add a new params object by using the setter:
             >>> state.params = {"theorist": {"n_epochs": 111}}
             >>> state.params  # doctest: +ELLIPSIS
             {'theorist': {'n_epochs': 111}}

             ... or by calling "update" directly:
             >>> state.update({"theorist": {"n_epochs": 222}}, kind=ResultKind.PARAMS)

             That adds the metadata object to the .data attribute:
             >>> state.data  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
             [Result(data={'theorist': {'n_epochs': 111}}, kind=ResultKind.PARAMS),
              Result(data={'theorist': {'n_epochs': 222}}, kind=ResultKind.PARAMS)]

              If we access the params property, we get the second one back:
              >>> state.params
              {'theorist': {'n_epochs': 222}}

        """
        try:
            p = self._get_last(kind={ResultKind.PARAMS}).data
        except StopIteration:
            p = dict()
        return p

    @params.setter
    def params(self, value):
        self.update(value, kind=ResultKind.PARAMS)

    @property
    def conditions(self) -> List[ArrayLike]:
        """
        Get all the results of kind "CONDITION"

        Examples:
            Initially, we get an empty list back:
            >>> state = CycleState()
            >>> state.conditions
            []

            We can add new conditions by using the `.update` method and specifying the kind as
            `"CONDITION"`:
            >>> import numpy as np
            >>> state.update(np.array([11,12,13]), kind="CONDITION")
            >>> state.conditions
            [array([11, 12, 13])]

            When we add multiple conditions, we get them all back:
            >>> state.update(np.array([21,22,23]), kind="CONDITION")
            >>> state.update(np.array([31,32,33]), kind="CONDITION")
            >>> state.conditions
            [array([11, 12, 13]), array([21, 22, 23]), array([31, 32, 33])]

        """
        return self._list_data(
            self._filter_result(self.data, kind={ResultKind.CONDITION})
        )

    @property
    def observations(self) -> List[ArrayLike]:
        """
        Get all the results of kind "OBSERVATION"

        Examples:
            Initially, we get an empty list back:
            >>> state = CycleState()
            >>> state.observations
            []

            We can add new conditions by using the `.update` method and specifying the kind as
            `"OBSERVATION"`:
            >>> import numpy as np
            >>> state.update(np.array([11,12,13]), kind="OBSERVATION")
            >>> state.observations
            [array([11, 12, 13])]

            When we add multiple conditions, we get them all back:
            >>> state.update(np.array([21,22,23]), kind="OBSERVATION")
            >>> state.update(np.array([31,32,33]), kind="OBSERVATION")
            >>> state.observations
            [array([11, 12, 13]), array([21, 22, 23]), array([31, 32, 33])]

        """
        return self._list_data(
            self._filter_result(self.data, kind={ResultKind.OBSERVATION})
        )

    @property
    def theories(self) -> List[BaseEstimator]:
        """
        Get all the results of kind "THEORY"

        Examples:
            Initially, we get an empty list back:
            >>> state = CycleState()
            >>> state.theories
            []

            We can add new conditions by using the `.update` method and specifying the kind as
            `"THEORY"`:
            >>> from sklearn.linear_model import LinearRegression
            >>> import numpy as np
            >>> theory_1 = LinearRegression().fit(np.array([[11,12,13]]), np.array([[111,112,113]]))
            >>> state.update(theory_1, kind="THEORY")
            >>> state.theories
            [LinearRegression()]

            When we add multiple theories, we get them all back:
            >>> theory_2 = LinearRegression().fit(np.array([[21,22,23]]), np.array([[121,122,123]]))
            >>> state.update(theory_2, kind="THEORY")
            >>> state.theories
            [LinearRegression(), LinearRegression()]
        """
        return self._list_data(self._filter_result(self.data, kind={ResultKind.THEORY}))

    @property
    def results(self) -> List[Result]:
        """
        Get all the CONDITION, OBSERVATION and THEORY result objects

        Examples:
            First, we create an empty state:
            >>> state = CycleState()

            Initially, `.results` is empty:
            >>> state.results
            []

            Usually there will be a metadata entry:
            >>> state.update("0 - metadata", kind="METADATA")

            We add a series of results in order:
            >>> state.update("1 - theory", kind="THEORY")
            >>> state.update("2 - condition", kind="CONDITION")
            >>> state.update("3 - observation", kind="OBSERVATION")
            >>> state.update("4 - theory", kind="THEORY")
            >>> state.update("5 - theory", kind="THEORY")
            >>> state.update("6 - observation", kind="THEORY")

            `.results` now includes all the results excluding the metadata:
            >>> state.results  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='1 - theory', kind=ResultKind.THEORY),
             Result(data='2 - condition', kind=ResultKind.CONDITION),
             Result(data='3 - observation', kind=ResultKind.OBSERVATION),
             Result(data='4 - theory', kind=ResultKind.THEORY),
             Result(data='5 - theory', kind=ResultKind.THEORY),
             Result(data='6 - observation', kind=ResultKind.THEORY)]

            If we add a PARAMS object, this is also ignored:
            >>> state.update({"some": "params"}, kind="PARAMS")
            >>> state.results  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='1 - theory', kind=ResultKind.THEORY),
            ...
            Result(data='6 - observation', kind=ResultKind.THEORY)]

        """
        return list(
            self._filter_result(
                self.data,
                kind={ResultKind.CONDITION, ResultKind.OBSERVATION, ResultKind.THEORY},
            )
        )

    def _get_last(self, kind):
        results_new_to_old = reversed(self.data)
        last_of_kind = next(self._filter_result(results_new_to_old, kind=kind))
        return last_of_kind

    @staticmethod
    def _filter_result(result_sequence: Sequence[Result], kind: set[ResultKind]):
        return filter(lambda r: r.kind in kind, result_sequence)

    @staticmethod
    def _list_data(result_sequence: Sequence[Result]):
        return list(r.data for r in result_sequence)


@dataclass(frozen=True)
class Result:
    """Container class for data and metadata."""

    data: Optional[Any]
    kind: Optional[ResultKind]


class ResultKind(Enum):
    """Kinds of results which can be held in the Result object"""

    CONDITION = "CONDITION"
    OBSERVATION = "OBSERVATION"
    THEORY = "THEORY"
    PARAMS = "PARAMS"
    METADATA = "METADATA"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class SupportsResultSequence(Protocol):
    data: Sequence[Result]


class SupportsConditionsObservationsTheories(Protocol):
    results: Sequence[Result]
