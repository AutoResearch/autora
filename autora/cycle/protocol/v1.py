""" First version of Protocols for AER. """
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Protocol, Sequence, TypeVar

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

T = TypeVar("T")


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[Any]


class Planner(Protocol):
    """A Callable which, based on some state, returns an Executor."""

    def __call__(self, __state, __executor_collection) -> Executor:
        ...


class Executor(Protocol):
    """A Callable which, given some state, returns an updated state."""

    def __call__(self, __state: T) -> T:
        ...


class SupportsFullCycle(Protocol):
    """An object which has a single executor that runs the full AER cycle."""

    full_cycle: Executor


class SupportsExperimentalistExperimentRunnerTheorist(Protocol):
    """Supports methods for the experimentalist, experiment runner and theorist of AER."""

    experimentalist: Executor
    experiment_runner: Executor
    theorist: Executor


class SupportsHistory(Protocol):
    history: Sequence[SupportsDataKind]

    def filter_by(self: T, **kwargs) -> T:
        ...


class SupportsConditionsObservationsTheories(Protocol):
    conditions: Sequence[ArrayLike]
    observations: Sequence[ArrayLike]
    theories: Sequence[BaseEstimator]


class SupportsMetadataParams(Protocol):
    metadata: VariableCollection
    params: Dict


class SupportsUpdate(Protocol):
    """Supports updating an immutable object and returning a new copy with updated values."""

    def update(self: T, **kwargs) -> T:
        ...


class SupportsCycleState(
    SupportsConditionsObservationsTheories,
    SupportsMetadataParams,
    SupportsUpdate,
    Protocol,
):
    ...

    def __init__(self, **kwargs) -> None:
        ...


class SupportsCycleStateHistory(
    SupportsConditionsObservationsTheories,
    SupportsMetadataParams,
    SupportsHistory,
    SupportsUpdate,
    Protocol,
):
    ...


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
