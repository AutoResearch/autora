import random
from itertools import product
from typing import List

import numpy as np

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


def random_pool(*args, n=1, duplicates=True):
    """
    Creates combinations from lists of discrete values using random selection.
    Args:
        *args: m lists of discrete values. One value will be sampled from each list.
        n: Number of samples to sample
        duplicates: Boolean if duplicate value are allowed.

    """
    l_samples = []
    # Create list of pools of values sample from
    pools = [tuple(pool) for pool in args]

    # Check to ensure infinite search won't occur if duplicates not allowed
    if not duplicates:
        l_pool_len = [len(set(s)) for s in pools]
        n_combinations = np.product(l_pool_len)
        try:
            assert n <= n_combinations
        except AssertionError:
            raise AssertionError(
                f"Number to sample n({n}) is larger than the number "
                f"of unique combinations({n_combinations})."
            )

    # Random sample from the pools until n is met
    while len(l_samples) < n:
        l_samples.append(tuple(map(random.choice, pools)))
        if not duplicates:
            l_samples = [*set(l_samples)]

    return iter(l_samples)
