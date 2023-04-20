import logging
import os
import pathlib
from typing import Optional

import typer
import yaml

from autora.controller import Controller
from autora.theorist.__main__ import _configure_logger

_logger = logging.getLogger(__name__)


def main(
    manager: pathlib.Path=typer.Argument(..., help="Manager path"),
    directory: pathlib.Path=typer.Argument(..., help="Directory path"),
    step_name: Optional[str] = typer.Argument(None, help="Name of step"),
    verbose: bool = typer.Option(False, help="Turns on info logging level."),
    debug: bool = typer.Option(False, help="Turns on debug logging level."),
):
    _logger.debug("initializing")
    _configure_logger(debug, verbose)
    controller_ = _load_manager(manager)

    _logger.debug(f"loading manager state from {directory=}")
    controller_.load(directory)

    if step_name is not None:
        controller_ = _set_next_step_name(controller_, step_name)

    _logger.info("running next step")
    next(controller_)

    _logger.debug(f"last result: {controller_.state.history[-1]}")

    _logger.info("writing out results")
    controller_.dump(directory)

    return


def _load_manager(path: pathlib.Path) -> Controller:
    _logger.debug(f"_load_manager: loading from {path=} (currently in {os.getcwd()})")
    with open(path, "r") as f:
        controller_ = yaml.load(f, yaml.Loader)
    assert isinstance(
        controller_, Controller
    ), f"controller type {type(controller_)=} unsupported"
    return controller_


def _set_next_step_name(controller: Controller, step_name: str):
    _logger.info(f"setting next {step_name=}")
    controller.planner = lambda _: step_name
    return controller


if __name__ == "__main__":
    typer.run(main)
