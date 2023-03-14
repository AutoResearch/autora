from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence

from numpy._typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


class ResultKind(Enum):
    CONDITION = "CONDITION"
    OBSERVATION = "OBSERVATION"
    THEORY = "THEORY"
    PARAMS = "PARAMS"
    METADATA = "METADATA"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


@dataclass(frozen=True)
class Result:
    data: Any
    kind: ResultKind


@dataclass
class CycleState:
    """An object passed between and updated by Executors."""

    data: List[Result] = field(default_factory=list)

    def __init__(
        self,
        metadata: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        conditions: Optional[Sequence[ArrayLike]] = None,
        observations: Optional[Sequence[ArrayLike]] = None,
        theories: Optional[Sequence[BaseEstimator]] = None,
        data: Optional[Sequence[Result]] = None,
    ):
        if data is not None:
            self.data = list(data)
        else:
            self.data = []

        for seq, kind in [
            (conditions, ResultKind.CONDITION),
            (observations, ResultKind.OBSERVATION),
            (theories, ResultKind.THEORY),
        ]:
            if seq is not None:
                for i in seq:
                    self.update(i, kind=kind)

        if metadata is not None:
            self.metadata = metadata

        if params is not None:
            self.params = params

    def update(self, value, kind):
        self.data.append(Result(value, ResultKind(kind)))

    @property
    def metadata(self) -> VariableCollection:
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
        return self._list_data(
            self._filter_result(self.data, kind={ResultKind.CONDITION})
        )

    @property
    def observations(self) -> List[ArrayLike]:
        return self._list_data(
            self._filter_result(self.data, kind={ResultKind.OBSERVATION})
        )

    @property
    def theories(self) -> List[BaseEstimator]:
        return self._list_data(self._filter_result(self.data, kind={ResultKind.THEORY}))

    @property
    def results(self) -> List[Result]:
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


class SupportsResultSequence(Protocol):
    data: Sequence[Result]


class SupportsConditionsObservationsTheories(Protocol):
    results: Sequence[Result]
