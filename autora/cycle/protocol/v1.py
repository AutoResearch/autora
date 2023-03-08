from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

from autora.cycle.result import Result, ResultCollection


@runtime_checkable
class Cycle(Protocol):
    state: ResultCollection
    params_resolved: Dict
    experimentalist: Executor
    experiment_runner: Executor
    theorist: Executor


@runtime_checkable
class Planner(Protocol):
    def __call__(self, cycle: Cycle) -> Executor:
        ...


@runtime_checkable
class Executor(Protocol):
    def __call__(self, cycle: Cycle) -> Result:
        ...
