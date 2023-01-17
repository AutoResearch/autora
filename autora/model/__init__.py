""" Models for testing within the AutoRA framework. """
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

    def model_v0(self):
        assert self.metadata is not None
        return self.metadata(), self.data, self.synthetic_experiment_runner

    def plotter_v0(self):
        return self.plotter, self.name


class ModelInventory(UserDict):
    pass


_INVENTORY = ModelInventory()


def register_model(
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


def retrieve_model(id, kind: Literal["model", "plotter"], version="v0"):
    entry: Model = _INVENTORY[id]
    if version == "v0":
        if kind == "model":
            return entry.model_v0()
        elif kind == "plotter":
            return entry.plotter_v0()
