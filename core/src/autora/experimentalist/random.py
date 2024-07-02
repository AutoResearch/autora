from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.variable import ValueType, VariableCollection


def pool(
    variables: VariableCollection,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    """
    Create a sequence of conditions randomly sampled from independent variables.

    Args:
        variables: the description of all the variables in the AER experiment.
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: the generated conditions as a dataframe

    Examples:
        >>> from autora.state import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        With one independent variable "x", and some allowed_values we get some of those values
        back when running the experimentalist:
        >>> pool(
        ...     VariableCollection(
        ...         independent_variables=[Variable(name="x", allowed_values=range(10))
        ... ]), random_state=1)
           x
        0  4
        1  5
        2  7
        3  9
        4  0


        ... with one independent variable "x", and a value_range,
        we get a sample of the range back when running the experimentalist:
        >>> pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", value_range=(-5, 5))
        ... ]), random_state=1)
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> pool(VariableCollection(independent_variables=[Variable(name="x")]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...

        With two independent variables, we get independent samples on both axes:
        >>> pool(VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=range(1, 5)),
        ...         Variable(name="x2", allowed_values=range(1, 500)),
        ... ]), num_samples=10, replace=True, random_state=1)
           x1   x2
        0   2  434
        1   3  212
        2   4  137
        3   4  414
        4   1  129
        5   1  205
        6   4  322
        7   4  275
        8   1   43
        9   2   14

        If any of the variables have unspecified allowed_values, we get an error:
        >>> pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...


        We can specify arrays of allowed values:

        >>> pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]), random_state=1)
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0


    """
    rng = np.random.default_rng(random_state)

    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(
                iv.allowed_values, size=num_samples, replace=replace
            )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_samples)

        else:
            raise ValueError(
                "allowed_values or [value_range and type==REAL] needs to be set for "
                "%s" % (iv)
            )

    return pd.DataFrame(raw_conditions)


random_pool = pool
"""Alias for `pool`"""


def sample(
    conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
    num_samples: int = 1,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> pd.DataFrame:
    """
    Take a random sample from some input conditions.

    Args:
        conditions: the conditions to sample from
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a Result object with a field `conditions` containing a DataFrame of the sampled
    conditions

    Examples:
        From a pd.DataFrame:
        >>> import pandas as pd
        >>> sample(
        ...     pd.DataFrame({"x": range(100, 200)}), num_samples=5, random_state=180)
              x
        67  167
        71  171
        64  164
        63  163
        96  196

        From a list (returns a DataFrame):
        >>> sample(range(1000), num_samples=5, random_state=180)
               0
        270  270
        908  908
        109  109
        331  331
        978  978
    """
    conditions_ = pd.DataFrame(conditions)
    return pd.DataFrame.sample(
        conditions_, random_state=random_state, n=num_samples, replace=replace
    )


random_sample = sample
"""Alias for `sample`"""
