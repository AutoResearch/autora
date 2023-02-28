from typing import Callable, Protocol, runtime_checkable

from autora.cycle.result import ResultCollection


@runtime_checkable
class Cycle(Protocol):
    state: ResultCollection

    def run_experimentalist(self):
        ...

    def run_experiment_runner(self):
        ...

    def run_theorist(self):
        ...


@runtime_checkable
class Planner(Protocol):
    def __call__(self, cycle: Cycle) -> Callable:
        ...
