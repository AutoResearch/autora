import importlib
import logging
import pathlib
from typing import Optional

import typer
from typing_extensions import Annotated

from autora.serializer import (
    SupportedSerializer,
    default_serializer,
    dump_state,
    load_state,
)

_logger = logging.getLogger(__name__)


def main(
    fully_qualified_function_name: Annotated[
        str,
        typer.Argument(
            help="Fully qualified name of the function to load, like `module.function`"
        ),
    ],
    in_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to a file with the initial state"),
    ] = None,
    in_loader: Annotated[
        SupportedSerializer,
        typer.Option(
            help="(de)serializer to load the data",
        ),
    ] = default_serializer,
    out_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to output the final state"),
    ] = None,
    out_dumper: Annotated[
        SupportedSerializer,
        typer.Option(
            help="serializer to save the data",
        ),
    ] = default_serializer,
    verbose: Annotated[bool, typer.Option(help="Turns on info logging level.")] = False,
    debug: Annotated[bool, typer.Option(help="Turns on debug logging level.")] = False,
):
    """Run an arbitrary function on an optional input State object and save the output."""
    _configure_logger(debug, verbose)
    starting_state = load_state(in_path, in_loader)
    _logger.info(f"Starting State: {starting_state}")
    function = _load_function(fully_qualified_function_name)
    ending_state = function(starting_state)
    _logger.info(f"Ending State: {ending_state}")
    dump_state(ending_state, out_path, out_dumper)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _load_function(fully_qualified_function_name: str):
    """Load a function by its fully qualified name, `module.function_name`"""
    _logger.debug(f"_load_function: Loading function {fully_qualified_function_name}")
    module_name, function_name = fully_qualified_function_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    _logger.debug(f"_load_function: Loaded function {function} from {module}")
    return function


if __name__ == "__main__":
    typer.run(main)
