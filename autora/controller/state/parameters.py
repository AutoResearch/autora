""" Functions for handling cycle-state-dependent parameters. """
from __future__ import annotations

import copy
import logging
from typing import Dict, Mapping

import numpy as np

from autora.controller.protocol import SupportsControllerState
from autora.utils.dictionary import LazyDict

_logger = logging.getLogger(__name__)


def _get_state_dependent_properties(state: SupportsControllerState):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> from autora.controller.state import Snapshot
        >>> state_dependent_properties = _get_state_dependent_properties(Snapshot())

        ... but it will raise an exception if a value isn't yet available when we try to use it
        >>> state_dependent_properties["%models[-1]%"] # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: list index out of range

        Nevertheless, we can iterate through its keys no problem:
        >>> [key for key in state_dependent_properties.keys()] # doctest: +NORMALIZE_WHITESPACE
        ['%observations.ivs[-1]%', '%observations.dvs[-1]%', '%observations.ivs%',
        '%observations.dvs%', '%models[-1]%', '%models%']

    """

    n_ivs = len(state.variables.independent_variables)
    n_dvs = len(state.variables.dependent_variables)
    state_dependent_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: np.array(state.observations[-1])[
                :, 0:n_ivs
            ],
            "%observations.dvs[-1]%": lambda: np.array(state.observations[-1])[
                :, n_ivs:
            ],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + list(state.observations)
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(state.observations)[:, n_ivs:],
            "%models[-1]%": lambda: state.models[-1],
            "%models%": lambda: state.models,
        }
    )
    return state_dependent_property_dict


def _resolve_properties(parameters: Dict, state_dependent_properties: Mapping):
    """
    Resolve state-dependent properties inside a nested dictionary.

    In this context, a state-dependent-property is a string which is meant to be replaced by its
    updated, current value before the dictionary is used. A state-dependent property might be
    something like "the last theorist available" or "all the experimental results until now".

    Args:
        parameters: a (nested) dictionary of keys and values, where some values might be
            "cycle property names"
        state_dependent_properties: a dictionary of "property names" and their "real values"

    Returns: a (nested) dictionary where "property names" are replaced by the "real values"

    Examples:

        >>> parameters_0 = {"key": "%foo%"}
        >>> cycle_properties_0 = {"%foo%": 180}
        >>> _resolve_properties(parameters_0,cycle_properties_0)
        {'key': 180}

        >>> parameters_1 = {"key": "%bar%", "nested_dict": {"inner_key": "%foobar%"}}
        >>> cycle_properties_1 = {"%bar%": 1, "%foobar%": 2}
        >>> _resolve_properties(parameters_1,cycle_properties_1)
        {'key': 1, 'nested_dict': {'inner_key': 2}}

        >>> parameters_2 = {"key": "baz"}
        >>> _resolve_properties(parameters_2,cycle_properties_1)
        {'key': 'baz'}

    """
    parameters_ = copy.copy(parameters)
    for key, value in parameters_.items():
        if isinstance(value, dict):
            parameters_[key] = _resolve_properties(value, state_dependent_properties)
        elif isinstance(value, str) and (
            value in state_dependent_properties
        ):  # value is a key in the cycle_properties dictionary
            parameters_[key] = state_dependent_properties[value]
        else:
            _logger.debug(f"leaving {parameters=} unchanged")

    return parameters_


def resolve_state_parameters(parameters: Dict, state: SupportsControllerState) -> Dict:
    """
    Returns the `parameters` attribute of the input, with `cycle properties` resolved.

    Examples:
        >>> from autora.controller.state import History
        >>> parameters = {"experimentalist": {"source": "%models[-1]%"}}
        >>> s = History(models=["the first model", "the second model"])
        >>> resolve_state_parameters(parameters, s)
        {'experimentalist': {'source': 'the second model'}}

    """
    state_dependent_properties = _get_state_dependent_properties(state)
    resolved_parameters = _resolve_properties(parameters, state_dependent_properties)
    return resolved_parameters
