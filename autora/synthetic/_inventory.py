from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

from autora.variable import VariableCollection


@dataclass
class SyntheticDataCollection:
    """Represents a synthetic experiment setup, including domain."""

    id: str
    name: Optional[str]
    metadata_callable: Optional[Callable[[], VariableCollection]]
    data_callable: Optional[Callable]
    synthetic_experiment_runner: Optional[Callable]
    plotter: Optional[Callable]


_INVENTORY: Dict[str, SyntheticDataCollection] = dict()
""" An dictionary of example datasets which can be used to test different functionality."""


def register(
    id,
    name=None,
    metadata_callable=None,
    synthetic_experiment_runner=None,
    data_callable=None,
    plotter=None,
):
    new_model = SyntheticDataCollection(
        id=id,
        name=name,
        metadata_callable=metadata_callable,
        synthetic_experiment_runner=synthetic_experiment_runner,
        data_callable=data_callable,
        plotter=plotter,
    )
    _INVENTORY[id] = new_model


def retrieve(
    id, kind: Optional[Literal["full:v0", "model:v0", "plotter:v0"]] = "full:v0"
):
    entry: SyntheticDataCollection = _INVENTORY[id]

    if kind == "full:v0":
        return entry
    if kind == "model:v0":
        assert entry.metadata_callable is not None
        return (
            entry.metadata_callable(),
            entry.data_callable,
            entry.synthetic_experiment_runner,
        )
    elif kind == "plotter:v0":
        return entry.plotter, entry.name
