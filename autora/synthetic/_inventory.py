from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol

from autora.variable import VariableCollection


@dataclass
class SyntheticExperimentCollection:
    """Represents a synthetic experiment setup, including domain."""

    name: Optional[str] = field(default=None)
    params: Optional[Dict] = field(default=None)
    metadata: Optional[VariableCollection] = field(default=None)
    domain: Optional[Callable] = field(default=None)
    experiment: Optional[Callable] = field(default=None)
    ground_truth: Optional[Callable] = field(default=None)
    plotter: Optional[Callable] = field(default=None)


class SyntheticExperimentClosure(Protocol):
    def __call__(self, *args, **kwargs) -> SyntheticExperimentCollection:
        ...


_INVENTORY: Dict[str, SyntheticExperimentClosure] = dict()
""" An dictionary of example datasets which can be used to test different functionality."""


def register(id_: str, closure: SyntheticExperimentClosure):
    _INVENTORY[id_] = closure


def retrieve(id_: str, **params):
    closure: SyntheticExperimentClosure = _INVENTORY[id_]
    evaluated_closure = closure(**params)
    return evaluated_closure
