from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

from autora.variable import VariableCollection


@dataclass
class SyntheticDataCollection:
    """Represents a synthetic experiment setup, including domain."""

    id_: str
    name: Optional[str]
    metadata: Optional[VariableCollection]
    domain: Optional[Callable]
    experiment: Optional[Callable]
    ground_truth: Optional[Callable]
    plotter: Optional[Callable]


_INVENTORY: Dict[str, SyntheticDataCollection] = dict()
""" An dictionary of example datasets which can be used to test different functionality."""


def register(
    id_,
    name=None,
    metadata=None,
    domain=None,
    experiment=None,
    ground_truth=None,
    plotter=None,
):
    new_model = SyntheticDataCollection(
        id_=id_,
        name=name,
        metadata=metadata,
        domain=domain,
        experiment=experiment,
        ground_truth=ground_truth,
    )
    _INVENTORY[id_] = new_model


def retrieve(
    id_: str, kind: Optional[Literal["full:v0", "model:v0", "plotter:v0"]] = "full:v0"
):
    entry: SyntheticDataCollection = _INVENTORY[id_]

    if kind == "full:v0":
        return entry
    if kind == "model:v0":
        assert entry.metadata is not None
        assert entry.domain is not None
        assert entry.ground_truth is not None
        domain = entry.domain()
        ground_truth_observations = entry.ground_truth(domain)
        return (
            entry.metadata,
            (domain, ground_truth_observations),
            entry.experiment,
        )
    elif kind == "plotter:v0":
        return entry.plotter, entry.name
