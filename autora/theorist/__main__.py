import importlib
import logging
from typing import Type

import typer
from sklearn.base import BaseEstimator

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


def main(regressor: str, verbose: bool = False, debug: bool = False):

    if verbose:
        logging.basicConfig(level=logging.INFO)
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    regressor_class = import_class(regressor)

    print(f"{regressor}: {regressor_class}")


if __name__ == "__main__":
    typer.run(main)
