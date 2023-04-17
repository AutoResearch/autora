from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Set, TypeVar, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

State = TypeVar("State")


class RecordKind(str, Enum):
    """
    Kinds of results which can be held in the Record object.

    Examples:
        >>> RecordKind.EXPERIMENT is RecordKind.EXPERIMENT
        True

        >>> RecordKind.EXPERIMENT is RecordKind.VARIABLES
        False

        >>> RecordKind.EXPERIMENT == "EXPERIMENT"
        True

        >>> RecordKind.EXPERIMENT == "VARIABLES"
        False

        >>> RecordKind.EXPERIMENT in {RecordKind.EXPERIMENT, RecordKind.PARAMETERS}
        True

        >>> RecordKind.VARIABLES in {RecordKind.EXPERIMENT, RecordKind.PARAMETERS}
        False
    """

    VARIABLES = "VARIABLES"
    PARAMETERS = "PARAMETERS"
    EXPERIMENT = "EXPERIMENT"
    OBSERVATION = "OBSERVATION"
    MODEL = "MODEL"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[RecordKind]


class SupportsControllerStateFields(Protocol):
    """Support representing snapshots of a controller state as mutable fields."""

    variables: VariableCollection
    parameters: Dict
    experiments: Sequence[ArrayLike]
    observations: Sequence[ArrayLike]
    models: Sequence[BaseEstimator]

    def update(self: State, **kwargs) -> State:
        ...


class SupportsControllerStateProperties(Protocol):
    """Support representing snapshots of a controller state as immutable properties."""

    def update(self: State, **kwargs) -> State:
        ...

    @property
    def variables(self) -> VariableCollection:
        ...

    @property
    def parameters(self) -> Dict:
        ...

    @property
    def experiments(self) -> Sequence[ArrayLike]:
        ...

    @property
    def observations(self) -> Sequence[ArrayLike]:
        ...

    @property
    def models(self) -> Sequence[BaseEstimator]:
        ...


SupportsControllerState = Union[
    SupportsControllerStateFields, SupportsControllerStateProperties
]


class SupportsControllerStateHistory(SupportsControllerStateProperties, Protocol):
    """Represents controller state as a linear sequence of entries."""

    def filter_by(self: State, kind: Optional[Set[Union[str, RecordKind]]]) -> State:
        ...

    @property
    def history(self) -> Sequence[SupportsDataKind]:
        ...


class Executor(Protocol):
    """A Callable which, given some state, and some parameters, returns an updated state."""

    def __call__(self, __state: State, __params: Dict) -> State:
        ...


ExecutorName = TypeVar("ExecutorName", bound=str)

ExecutorCollection = Mapping[ExecutorName, Executor]
