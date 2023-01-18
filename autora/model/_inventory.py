from collections import UserDict
from dataclasses import dataclass
from typing import Callable, Literal, Optional


@dataclass
class Model:
    id: str
    name: Optional[str]
    metadata: Optional[Callable]
    synthetic_experiment_runner: Optional[Callable]
    data: Optional[Callable]
    plotter: Optional[Callable]


class ModelInventory(UserDict):
    pass


_INVENTORY = ModelInventory()


def register(
    id,
    name=None,
    metadata=None,
    synthetic_experiment_runner=None,
    data=None,
    plotter=None,
):
    new_model = Model(
        id=id,
        name=name,
        metadata=metadata,
        synthetic_experiment_runner=synthetic_experiment_runner,
        data=data,
        plotter=plotter,
    )
    _INVENTORY[id] = new_model


def retrieve(
    id, kind: Optional[Literal["full:v0", "model:v0", "plotter:v0"]] = "full:v0"
):
    entry: Model = _INVENTORY[id]

    if kind == "full:v0":
        return entry
    if kind == "model:v0":
        assert entry.metadata is not None
        return entry.metadata(), entry.data, entry.synthetic_experiment_runner
    elif kind == "plotter:v0":
        return entry.plotter, entry.name
