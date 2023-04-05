from itertools import product
from typing import List

from autora.variable import IV


def grid_pool(ivs: List[IV]):
    """Creates exhaustive pool from discrete values using a Cartesian product of sets"""
    # Get allowed values for each IV
    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"gridsearch_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)
