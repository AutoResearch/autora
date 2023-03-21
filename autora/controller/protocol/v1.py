from enum import Enum
from typing import Any, Dict, Optional, Protocol, Sequence, TypeVar, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

State = TypeVar("State")


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


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[ResultKind]


class SupportsControllerStateFields(Protocol):
    """Support representing snapshots of a controller state as mutable fields."""

    def __init__(
        self,
        metadata: Optional[VariableCollection],
        params: Optional[Dict],
        conditions: Optional[Sequence[ArrayLike]],
        observations=Optional[Sequence[ArrayLike]],
        theories=Optional[Sequence[ArrayLike]],
    ) -> None:
        ...

    metadata: VariableCollection
    params: Dict
    conditions: Sequence[ArrayLike]
    observations: Sequence[ArrayLike]
    theories: Sequence[BaseEstimator]

    def update(self: State, **kwargs) -> State:
        ...


class SupportsControllerStateProperties(Protocol):
    """Support representing snapshots of a controller state as immutable properties."""

    def __init__(
        self,
        metadata: Optional[VariableCollection],
        params: Optional[Dict],
        conditions: Optional[Sequence[ArrayLike]],
        observations=Optional[Sequence[ArrayLike]],
        theories=Optional[Sequence[ArrayLike]],
    ) -> None:
        ...

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

    def __init__(
        self,
        metadata: Optional[VariableCollection],
        params: Optional[Dict],
        conditions: Optional[Sequence[ArrayLike]],
        observations=Optional[Sequence[ArrayLike]],
        theories=Optional[Sequence[ArrayLike]],
        history=Optional[Sequence[SupportsDataKind]],
    ) -> None:
        ...

    def filter_by(self: State, **kwargs) -> State:
        ...

    @property
    def history(self) -> Sequence[SupportsDataKind]:
        ...


class Executor(Protocol):
    """A Callable which, given some state, returns an updated state."""

    def __call__(self, __state: State) -> State:
        ...


ExecutorCollection = Dict[str, Executor]
