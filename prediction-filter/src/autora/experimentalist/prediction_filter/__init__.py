"""
Example Experimentalist
"""
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def filter(
    conditions: Union[pd.DataFrame, np.ndarray],
    model: BaseEstimator,
    filter_function: Callable,
) -> pd.DataFrame:
    """
    Filter conditions based on the expected outcome io the mdeol

    Args:
        conditions: The pool to filter
        model: The model to make the prediction
        filter_function: A function that returns True if a prediciton should be included

    Returns:
        Filtered pool of experimental conditions

    Examples:
        >>> class ModelLinear:
        ...     def predict(self, X):
        ...         c_array = np.array(X)
        ...         return 2 * c_array + 1
        >>> model = ModelLinear()
        >>> model.predict(4)
        9

        For the filter function, be aware of the output type of the predict function. For example,
        here, we expect a list with a single entry
        >>> filter_fct = lambda x: 5 < x < 10
        >>> pool = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
        >>> filter(pool, model, filter_fct)
           x
        0  3
        1  4

        >>> filter_fct_2d = lambda x: 4 < x[0] + x[1] < 10
        >>> pool = np.array([[1, 0], [0, 1], [0, 1], [1 ,1], [2, 2]])
        >>> model.predict(pool)
        array([[3, 1],
               [1, 3],
               [1, 3],
               [3, 3],
               [5, 5]])

        >>> filter(pool, model, filter_fct_2d)
           0  1
        0  1  1

        >>> pool = pd.DataFrame({'x': [1, 0, 0, 1, 2], 'y': [0, 1, 1, 1, 2]})
        >>> model.predict(pool)
        array([[3, 1],
               [1, 3],
               [1, 3],
               [3, 3],
               [5, 5]])

        >>> filter(pool, model, filter_fct_2d)
           x  y
        0  1  1
    """
    _pred = model.predict(conditions)
    _filter = np.apply_along_axis(filter_function, 1, _pred)
    _filter = _filter.reshape(1, -1)

    new_conditions = conditions[list(_filter[0])]

    if isinstance(conditions, pd.DataFrame):
        new_conditions = pd.DataFrame(
            new_conditions, columns=conditions.columns
        ).reset_index(drop=True)
    else:
        new_conditions = pd.DataFrame(new_conditions)

    return new_conditions


prediction_filter = filter
