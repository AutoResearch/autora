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

from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


def main(
    variables: pathlib.Path,
    regressor: pathlib.Path,
    parameters: pathlib.Path,
    data: pathlib.Path,
    output: pathlib.Path,
    verbose: bool = False,
    debug: bool = False,
    overwrite: bool = False,
):
    # Initialization
    _configure_logger(debug, verbose)

    # Data Loading
    variables_ = _load_variables(variables)
    parameters_ = _load_parameters(parameters)
    regressor_ = _load_regressor(regressor)
    data_ = _load_data(data)

    # Fitting
    model = _fit_model(data_, parameters_, regressor_, variables_)

    # Writing results
    _dump_model(model, output, overwrite)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _load_variables(path: pathlib.Path) -> VariableCollection:
    _logger.debug(f"load_variables: loading from {path=}")
    variables_: VariableCollection
    with open(path, "r") as fv:
        variables_ = yaml.load(fv, yaml.Loader)
    assert isinstance(variables_, VariableCollection)
    return variables_


def _load_parameters(path: pathlib.Path) -> Dict:
    _logger.debug(f"load_parameters: loading from {path=}")
    with open(path, "r") as fp:
        parameters_: Optional[Dict] = yaml.load(fp, yaml.Loader)
        if parameters_ is None:
            parameters_ = dict()
    return parameters_


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
