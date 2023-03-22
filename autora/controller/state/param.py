""" Functions for handling cycle-state-dependent parameters. """
from __future__ import annotations

import copy
from typing import Dict, Mapping

import numpy as np

from autora.controller.protocol import SupportsControllerState
from autora.utils.dictionary import LazyDict


def _get_state_dependent_properties(state: SupportsControllerState):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> from autora.controller.state import Snapshot
        >>> state_dependent_properties = _get_state_dependent_properties(Snapshot())

        ... but it will raise an exception if a value isn't yet available when we try to use it
        >>> state_dependent_properties["%theories[-1]%"] # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: list index out of range

        Nevertheless, we can iterate through its keys no problem:
        >>> [key for key in state_dependent_properties.keys()] # doctest: +NORMALIZE_WHITESPACE
        ['%observations.ivs[-1]%', '%observations.dvs[-1]%', '%observations.ivs%',
        '%observations.dvs%', '%theories[-1]%', '%theories%']

    """

    n_ivs = len(state.metadata.independent_variables)
    n_dvs = len(state.metadata.dependent_variables)
    state_dependent_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: state.observations[-1][:, 0:n_ivs],
            "%observations.dvs[-1]%": lambda: state.observations[-1][:, n_ivs:],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + list(state.observations)
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(state.observations)[:, n_ivs:],
            "%theories[-1]%": lambda: state.theories[-1],
            "%theories%": lambda: state.theories,
        }
    )
    return state_dependent_property_dict


def _resolve_properties(params: Dict, state_dependent_properties: Mapping):
    """
    Resolve state-dependent properties inside a nested dictionary.

    In this context, a state-dependent-property is a string which is meant to be replaced by its
    updated, current value before the dictionary is used. A state-dependent property might be
    something like "the last theorist available" or "all the experimental results until now".

    Args:
        params: a (nested) dictionary of keys and values, where some values might be
            "cycle property names"
        state_dependent_properties: a dictionary of "property names" and their "real values"

    Returns: a (nested) dictionary where "property names" are replaced by the "real values"

    Examples:

        >>> params_0 = {"key": "%foo%"}
        >>> cycle_properties_0 = {"%foo%": 180}
        >>> _resolve_properties(params_0,cycle_properties_0)
        {'key': 180}

        >>> params_1 = {"key": "%bar%", "nested_dict": {"inner_key": "%foobar%"}}
        >>> cycle_properties_1 = {"%bar%": 1, "%foobar%": 2}
        >>> _resolve_properties(params_1,cycle_properties_1)
        {'key': 1, 'nested_dict': {'inner_key': 2}}

        >>> params_2 = {"key": "baz"}
        >>> _resolve_properties(params_2,cycle_properties_1)
        {'key': 'baz'}

    """
    params_ = copy.copy(params)
    for key, value in params_.items():
        if isinstance(value, dict):
            params_[key] = _resolve_properties(value, state_dependent_properties)
        elif isinstance(value, str) and (
            value in state_dependent_properties
        ):  # value is a key in the cycle_properties dictionary
            params_[key] = state_dependent_properties[value]
        else:
            pass  # no change needed

    return params_


def resolve_state_params(state: SupportsControllerState) -> Dict:
    """
    Returns the `params` attribute of the input, with `cycle properties` resolved.

    Examples:
        >>> from autora.controller.state import History
        >>> s = History(theories=["the first theory", "the second theory"],
        ...     params={"experimentalist": {"source": "%theories[-1]%"}})
        >>> resolve_state_params(s)
        {'experimentalist': {'source': 'the second theory'}}

    """
    state_dependent_properties = _get_state_dependent_properties(state)
    resolved_params = _resolve_properties(state.params, state_dependent_properties)
    return resolved_params
