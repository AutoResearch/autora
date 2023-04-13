import importlib
import logging
import pathlib
import pickle
import pprint
from typing import Dict, Optional, Type

import pandas as pd
import typer
import yaml
from pandas import DataFrame
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


def import_class(name: str) -> Type[BaseEstimator]:
    """
    Load a class from a module by name.

    Args:
        name:

    Examples:
        >>> import_class("sklearn.linear_model.LinearRegressor")

    """
    components = name.split(".")
    module_name, class_name = ".".join(components[:-1]), components[-1]
    _logger.info(f"loading {module_name=}, {class_name=}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def main(
    variables: pathlib.Path,
    parameters: pathlib.Path,
    regressor: str,
    data: pathlib.Path,
    output: pathlib.Path,
    verbose: bool = False,
    debug: bool = False,
    overwrite: bool = False,
):
    # Initialization
    configure_logger(debug, verbose)

    # Data Loading
    variables_ = load_variables(variables)
    parameters_ = load_parameters(parameters)
    regressor_class_ = load_regressor_class(regressor)
    data_ = load_data(data)

    # Fitting
    model = fit_model(data_, parameters_, regressor_class_, variables_)

    # Writing results
    dump_model(model, output, overwrite)

    return


def configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def load_variables(path: pathlib.Path) -> VariableCollection:
    _logger.debug(f"load_variables: loading from {path=}")
    variables_: VariableCollection
    with open(path, "r") as fv:
        variables_ = yaml.load(fv, yaml.Loader)
    assert isinstance(variables_, VariableCollection)
    return variables_


def load_parameters(path: pathlib.Path) -> Dict:
    _logger.debug(f"load_parameters: loading from {path=}")
    with open(path, "r") as fp:
        parameters_: Optional[Dict] = yaml.load(fp, yaml.Loader)
        if parameters_ is None:
            parameters_ = dict()
    return parameters_


def load_regressor_class(regressor):
    regressor_class = import_class(regressor)
    _logger.info(f"{regressor}: {regressor_class}")
    return regressor_class


def load_data(data: pathlib.Path) -> DataFrame:
    _logger.debug(f"load_data: loading from {data=}")
    with open(data, "r") as fd:
        data_: DataFrame = pd.read_csv(fd)
    return data_


def fit_model(data_, parameters_, regressor_class_, variables_):
    model = regressor_class_(**parameters_)
    X = data_[[v.name for v in variables_.independent_variables]]
    y = data_[[v.name for v in variables_.dependent_variables]]
    _logger.debug(f"fitting the regressor with X:\n{X}\nand y:\n{y}")
    model.fit(X, y)
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


def dump_model(model_, output, overwrite):
    if overwrite:
        mode = "wb"
        _logger.info(f"overwriting {output=} if it already exists")
    else:
        mode = "xb"
        _logger.info(f"writing to new file {output=}")
    with open(output, mode) as o:
        pickle.dump(model_, o)


if __name__ == "__main__":
    typer.run(main)
