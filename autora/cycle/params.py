from __future__ import annotations

import copy
from typing import Dict, Mapping

import numpy as np

from autora.cycle.state import SimpleCycleData
from autora.utils.dictionary import LazyDict


def _get_cycle_properties(data: SimpleCycleData):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> from autora.variable import VariableCollection
        >>> cycle_properties = _get_cycle_properties(SimpleCycleData(metadata=VariableCollection(),
        ...     conditions=[], observations=[], theories=[]))

        ... but it will raise an exception if a value isn't yet available when we try to use it
        >>> cycle_properties["%theories[-1]%"] # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: list index out of range

        Nevertheless, we can iterate through its keys no problem:
        >>> [key for key in cycle_properties.keys()] # doctest: +NORMALIZE_WHITESPACE
        ['%observations.ivs[-1]%', '%observations.dvs[-1]%', '%observations.ivs%',
        '%observations.dvs%', '%theories[-1]%', '%theories%']

    """

    n_ivs = len(data.metadata.independent_variables)
    n_dvs = len(data.metadata.dependent_variables)
    cycle_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: data.observations[-1][:, 0:n_ivs],
            "%observations.dvs[-1]%": lambda: data.observations[-1][:, n_ivs:],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + data.observations
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(data.observations)[:, n_ivs:],
            "%theories[-1]%": lambda: data.theories[-1],
            "%theories%": lambda: data.theories,
        }
    )
    return cycle_property_dict


def _resolve_cycle_properties(params: Dict, cycle_properties: Mapping):
    """
    Resolve "cycle properties" inside a nested dictionary.

    In this context, a "cycle property" is a string which is meant to be replaced by a
    different value before the dictionary is used.

    Args:
        params: a (nested) dictionary of keys and values, where some values might be
            "cycle property names"
        cycle_properties: a dictionary of "cycle property names" and their "real values"

    Returns: a (nested) dictionary where "cycle property names" are replaced by the "real values"

    Examples:

        >>> params_0 = {"key": "%foo%"}
        >>> cycle_properties_0 = {"%foo%": 180}
        >>> _resolve_cycle_properties(params_0, cycle_properties_0)
        {'key': 180}

        >>> params_1 = {"key": "%bar%", "nested_dict": {"inner_key": "%foobar%"}}
        >>> cycle_properties_1 = {"%bar%": 1, "%foobar%": 2}
        >>> _resolve_cycle_properties(params_1, cycle_properties_1)
        {'key': 1, 'nested_dict': {'inner_key': 2}}

    """
    params_ = copy.copy(params)
    for key, value in params_.items():
        if isinstance(value, dict):
            params_[key] = _resolve_cycle_properties(value, cycle_properties)
        elif (
            isinstance(value, str) and value in cycle_properties
        ):  # value is a key in the cycle_properties dictionary
            params_[key] = cycle_properties[value]
        else:
            pass  # no change needed

    return params_
