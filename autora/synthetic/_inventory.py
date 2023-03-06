from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from autora.variable import VariableCollection


@dataclass
class SyntheticExperimentCollection:
    """Represents a synthetic experiment setup, including domain."""

    name: Optional[str] = None
    params: Optional[Dict] = None
    metadata: Optional[VariableCollection] = None
    domain: Optional[Callable] = None
    experiment_runner: Optional[Callable] = None
    ground_truth: Optional[Callable] = None
    plotter: Optional[Callable] = None


class SyntheticExperimentClosure(Protocol):
    def __call__(self, *args, **kwargs) -> SyntheticExperimentCollection:
        ...


Inventory: Dict[str, SyntheticExperimentClosure] = dict()
""" An dictionary of example datasets which can be used to test different functionality."""


def register(id: str, closure: SyntheticExperimentClosure):
    Inventory[id] = closure


def retrieve(id: str, **params):
    closure: SyntheticExperimentClosure = Inventory[id]
    evaluated_closure = closure(**params)
    return evaluated_closure
