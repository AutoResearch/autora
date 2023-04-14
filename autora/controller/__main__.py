import logging
import pathlib

import typer
import yaml

from autora.controller import Controller
from autora.theorist.__main__ import _configure_logger

_logger = logging.getLogger(__name__)


def main(
    manager: pathlib.Path,
    directory: pathlib.Path,
    step_name: str,
    verbose: bool = False,
    debug: bool = False,
):
    _logger.debug("initializing")
    _configure_logger(debug, verbose)
    controller_ = _load_manager(manager)

    _logger.debug(f"loading manager state from {directory=}")
    controller_.load(directory)

    _logger.info(f"running {step_name=}")
    controller_.planner = lambda _: step_name
    next(controller_)

    _logger.info("writing out results")
    controller_.dump(directory)

    return


def _load_manager(path: pathlib.Path) -> Controller:
    _logger.debug(f"_load_manager: loading from {path=}")
    with open(path, "r") as f:
        controller_ = yaml.load(f, yaml.Loader)
    assert isinstance(
        controller_, Controller
    ), f"controller type {type(controller_)=} unsupported"
    return controller_


if __name__ == "__main__":
    typer.run(main)
