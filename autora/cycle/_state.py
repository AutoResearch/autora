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


@dataclass
class Result:
    data: Any
    kind: ResultKind


@dataclass
class CycleState:
    """An object passed between and updated by Executors."""

    result_sequence: List[Result] = field(default_factory=list)

    def __init__(
        self,
        metadata: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        conditions: Optional[Sequence[ArrayLike]] = None,
        observations: Optional[Sequence[ArrayLike]] = None,
        theories: Optional[Sequence[BaseEstimator]] = None,
        result_sequence: Optional[Sequence[Result]] = None,
    ):
        if result_sequence is not None:
            self.result_sequence = list(result_sequence)
        else:
            self.result_sequence = []

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

    def update(self, data, kind):
        self.result_sequence.append(Result(data, ResultKind(kind)))

    @property
    def metadata(self) -> VariableCollection:
        try:
            m = self._get_last_data_by_kind(kind={ResultKind.METADATA})
        except StopIteration:
            m = VariableCollection()
        return m

    @metadata.setter
    def metadata(self, value):
        self.update(value, kind=ResultKind.METADATA)

    @property
    def params(self) -> Dict:
        try:
            p = self._get_last_data_by_kind(kind={ResultKind.PARAMS})
        except StopIteration:
            p = dict()
        return p

    @params.setter
    def params(self, value):
        self.update(value, kind=ResultKind.PARAMS)

    @property
    def conditions(self) -> List[ArrayLike]:
        return list(
            self._filter_data_by_kind(self.result_sequence, kind={ResultKind.CONDITION})
        )

    @property
    def observations(self) -> List[ArrayLike]:
        return list(
            self._filter_data_by_kind(
                self.result_sequence, kind={ResultKind.OBSERVATION}
            )
        )

    @property
    def theories(self) -> List[BaseEstimator]:
        return list(
            self._filter_data_by_kind(self.result_sequence, kind={ResultKind.THEORY})
        )

    @property
    def conditions_observations_theories(self) -> List[Result]:
        return list(
            self._filter_result_by_kind(
                self.result_sequence,
                kind={ResultKind.CONDITION, ResultKind.OBSERVATION, ResultKind.THEORY},
            )
        )

    def _get_last_data_by_kind(self, kind):
        results_new_to_old = reversed(self.result_sequence)
        last_of_kind = next(self._filter_data_by_kind(results_new_to_old, kind=kind))
        return last_of_kind

    @staticmethod
    def _filter_data_by_kind(result_sequence: Sequence[Result], kind: set[ResultKind]):
        return (
            r.data
            for r in CycleState._filter_result_by_kind(result_sequence, kind=kind)
        )

    @staticmethod
    def _filter_result_by_kind(
        result_sequence: Sequence[Result], kind: set[ResultKind]
    ):
        return filter(lambda r: r.kind in kind, result_sequence)


class SupportsResultSequence(Protocol):
    result_sequence: Sequence[Result]


class SupportsConditionsObservationsTheories(Protocol):
    conditions_observations_theories: Sequence[Result]
