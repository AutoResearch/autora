from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, NamedTuple, Optional

from autora.variable import VariableCollection


class ResultKind(Enum):
    CONDITION = auto()
    OBSERVATION = auto()
    THEORY = auto()

    def __str__(self):
        return f"{self.name}"


class Result(NamedTuple):
    data: Optional[Any]
    kind: Optional[ResultKind]


@dataclass
class ResultCollection:
    metadata: VariableCollection
    data: List[Result]

    @property
    def conditions(self):
        return self._filter(ResultKind.CONDITION)

    @property
    def observations(self):
        return self._filter(ResultKind.OBSERVATION)

    @property
    def theories(self):
        return self._filter(ResultKind.THEORY)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Result:
        return self.data[index]

    def append(self, item: Result):
        self.data.append(item)
        return

    def _filter(self, kind: ResultKind):
        filtered_data = [
            c.data for c in self.data if ((c.kind is not None) and (c.kind == kind))
        ]
        return filtered_data
