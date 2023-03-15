from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[Any]


class Planner(Protocol):
    """A Callable which, based on some state, returns an Executor."""

    def __call__(self, state_or_history, executor_collection) -> Executor:
        ...


class Executor(Protocol):
    """A Callable which, given some state, returns 1 or more updates."""

    def __call__(self, state: Sequence[SupportsDataKind]) -> Sequence[SupportsDataKind]:
        ...


class SupportsFullCycle(Protocol):
    """An object which has a single executor that runs the full AER cycle."""

    full_cycle: Executor


class SupportsExperimentalistExperimentRunnerTheorist(Protocol):
    """Supports methods for the experimentalist, experiment runner and theorist of AER."""

    experimentalist: Executor
    experiment_runner: Executor
    theorist: Executor
