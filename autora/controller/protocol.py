from enum import Enum
from typing import Any, Dict, Optional, Protocol, Sequence, Set, TypeVar, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

State = TypeVar("State")


class ResultKind(str, Enum):
    """
    Kinds of results which can be held in the Result object.

    Examples:
        >>> ResultKind.EXPERIMENT is ResultKind.EXPERIMENT
        True

        >>> ResultKind.EXPERIMENT is ResultKind.VARIABLES
        False

        >>> ResultKind.EXPERIMENT == "EXPERIMENT"
        True

        >>> ResultKind.EXPERIMENT == "VARIABLES"
        False

        >>> ResultKind.EXPERIMENT in {ResultKind.EXPERIMENT, ResultKind.PARAMETERS}
        True

        >>> ResultKind.VARIABLES in {ResultKind.EXPERIMENT, ResultKind.PARAMETERS}
        False
    """

    EXPERIMENT = "EXPERIMENT"
    OBSERVATION = "OBSERVATION"
    THEORY = "THEORY"
    PARAMETERS = "PARAMETERS"
    VARIABLES = "VARIABLES"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[ResultKind]


class SupportsControllerStateFields(Protocol):
    """Support representing snapshots of a controller state as mutable fields."""

    metadata: VariableCollection
    params: Dict
    conditions: Sequence[ArrayLike]
    observations: Sequence[ArrayLike]
    theories: Sequence[BaseEstimator]

    def update(self: State, **kwargs) -> State:
        ...


class SupportsControllerStateProperties(Protocol):
    """Support representing snapshots of a controller state as immutable properties."""

    def update(self: State, **kwargs) -> State:
        ...

    @property
    def metadata(self) -> VariableCollection:
        ...

    @property
    def params(self) -> Dict:
        ...

    @property
    def conditions(self) -> Sequence[ArrayLike]:
        ...

    @property
    def observations(self) -> Sequence[ArrayLike]:
        ...

    @property
    def theories(self) -> Sequence[BaseEstimator]:
        ...


SupportsControllerState = Union[
    SupportsControllerStateFields, SupportsControllerStateProperties
]


class SupportsControllerStateHistory(SupportsControllerStateProperties, Protocol):
    """Represents controller state as a linear sequence of entries."""

    def filter_by(self: State, kind: Optional[Set[Union[str, ResultKind]]]) -> State:
        ...

    @property
    def history(self) -> Sequence[SupportsDataKind]:
        ...
