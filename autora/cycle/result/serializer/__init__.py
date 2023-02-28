from typing import Protocol, runtime_checkable

from autora.cycle.result import ResultCollection


@runtime_checkable
class ResultCollectionSerializer(Protocol):
    def load(self) -> ResultCollection:
        ...

    def dump(self, data: ResultCollection) -> None:
        ...
