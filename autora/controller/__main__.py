import logging
import pathlib
import pickle
import pprint
from typing import Dict, Optional

import pandas as pd
import typer
import yaml
from pandas import DataFrame
from sklearn.base import BaseEstimator

from autora.controller.protocol import SupportsControllerStateLoadDumpTarget
from autora.theorist.__main__ import _configure_logger

_logger = logging.getLogger(__name__)


def main(
    manager: pathlib.Path,
    directory: pathlib.Path,
    step: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
):
    # Initialize
    _configure_logger(debug, verbose)

    # Load data
    manager_ = _load_manager(manager)
    manager_.load(directory)

    manager_.state = manager_.state.update(params={"a": 1})

    # Run next step
    if step is not None:

        manager_.run_fn(step)

    # Write results
    manager_.dump(directory)

    return


def _load_manager(path: pathlib.Path) -> SupportsControllerStateLoadDumpTarget:
    _logger.debug(f"_load_manager: loading from {path=}")
    with open(path, "r") as f:
        manager_ = yaml.load(f, yaml.Loader)
    return manager_


def _load_regressor(path: pathlib.Path) -> BaseEstimator:
    with open(path, "r") as f:
        regressor_ = yaml.load(f, yaml.Loader)
    return regressor_


def _load_data(data: pathlib.Path) -> DataFrame:
    _logger.debug(f"load_data: loading from {data=}")
    with open(data, "r") as fd:
        data_: DataFrame = pd.read_csv(fd)
    return data_


def _fit_model(data, parameters, regressor, variables) -> BaseEstimator:
    model = regressor.set_params(**parameters)
    x = data[[v.name for v in variables.independent_variables]]
    y = data[[v.name for v in variables.dependent_variables]]
    _logger.debug(f"fitting the regressor with x's:\n{x}\nand y's:\n{y}")
    model.fit(x, y)
    try:
        _logger.info(
            f"fitted {model=}\nmodel.__dict__:" f"\n{pprint.pformat(model.__dict__)}"
        )
    except AttributeError:
        _logger.warning(
            f"fitted {model=} "
            f"model has no __dict__ attribute, so no results are shown"
        )
    return model


def _dump_model(model, output, overwrite):
    if overwrite:
        mode = "wb"
        _logger.info(f"overwriting {output=} if it already exists")
    else:
        mode = "xb"
        _logger.info(f"writing to new file {output=}")
    with open(output, mode) as o:
        pickle.dump(model, o)


if __name__ == "__main__":
    typer.run(main)
