from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric

from autora.utils.deprecation import deprecated_alias

AllowedMetrics = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "wminkowski",
    "seuclidean",
    "mahalanobis",
    "haversine",
    "hamming",
    "canberra",
    "braycurtis",
    "matching",
    "jaccard",
    "dice",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: int = 1,
    equality_distance: float = 0,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    This inequality experimentalist chooses from the pool of IV conditions according to their
    inequality with respect to a reference pool reference_conditions. Two IVs are considered
    equal if their distance is less than the equality_distance. The IVs chosen first are feed back
    into reference_conditions and are included in the summed equality calculation.

    Args:
        conditions: pool of IV conditions to evaluate inequality
        reference_conditions: reference pool of IV conditions
        num_samples: number of samples to select
        equality_distance: the distance to decide if two data points are equal.
        metric: inequality measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See `sklearn.metrics.DistanceMetric` for more details.

    Returns:
        Sampled pool

    Examples:
        The value 1 is not in the reference. Therefore it is choosen.
        >>> summed_inequality_sample([1, 2, 3], [2, 3, 4])
           0
        0  1

        The equality distance is set to 0.4. 1 and 1.3 are considered equal, so are 3 and 3.1.
        Therefore 2 is choosen.
        >>> summed_inequality_sample([1, 2, 3], [1.3, 2.7, 3.1], 1, .4)
           0
        0  2

        The value 3 appears least often in the reference.
        >>> summed_inequality_sample([1, 2, 3], [1, 1, 1, 2, 2, 2, 3, 3])
           0
        0  3

        The experimentalist "fills up" the reference array so the values are contributed evenly
        >>> summed_inequality_sample([1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 2, 2, 3, 3, 3], 3)
           0
        0  1
        1  3
        2  1

        The experimentalist samples without replacemnt!
        >>> summed_inequality_sample([1, 2, 3], [1, 1, 1], 3)
           0
        0  3
        1  2
        2  1

    """

    X = np.array(conditions)

    _reference_conditions = reference_conditions.copy()
    if isinstance(reference_conditions, pd.DataFrame):
        if set(conditions.columns) != set(reference_conditions.columns):
            raise Exception(
                f"Variable names {set(conditions.columns)} in conditions"
                f"and {set(reference_conditions.columns)} in allowed values don't match. "
            )

        _reference_conditions = _reference_conditions[conditions.columns]

    X_reference_conditions = np.array(_reference_conditions)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X_reference_conditions.ndim == 1:
        X_reference_conditions = X_reference_conditions.reshape(-1, 1)

    if X.shape[1] != X_reference_conditions.shape[1]:
        raise ValueError(
            f"conditions and reference_conditions must have the same number of columns.\n"
            f"conditions has {X.shape[1]} columns, "
            f"while reference_conditions has {X_reference_conditions.shape[1]} columns."
        )

    if X.shape[0] < num_samples:
        raise ValueError(
            f"conditions must have at least {num_samples} rows matching the number "
            f"of requested samples."
        )

    dist = DistanceMetric.get_metric(metric)

    # create a list to store the n conditions values with the highest inequality scores
    condition_pool_res = []
    # choose the canditate with the highest inequality score n-times
    for _ in range(num_samples):
        summed_equalities = []
        # loop over all IV values
        for row in X:

            # calculate the distances between the current row in matrix1
            # and all other rows in matrix2
            summed_equality = 0
            for reference_conditions_row in X_reference_conditions:
                distance = dist.pairwise([row, reference_conditions_row])[0, 1]
                summed_equality += distance > equality_distance

            # store the summed distance for the current row
            summed_equalities.append(summed_equality)

        # sort the rows in matrix1 by their summed distances
        X = X[np.argsort(summed_equalities)[::-1]]
        # append the first value of the sorted list to the result
        condition_pool_res.append(X[0])
        # add the chosen value to reference_conditions
        X_reference_conditions = np.append(X_reference_conditions, [X[0]], axis=0)
        # remove the chosen value from X
        X = X[1:]

    new_conditions = np.array(condition_pool_res[:num_samples])
    if isinstance(conditions, pd.DataFrame):
        new_conditions = pd.DataFrame(new_conditions, columns=conditions.columns)
    else:
        new_conditions = pd.DataFrame(new_conditions)
    return new_conditions


summed_inequality_sample = sample

summed_inequality_sampler = deprecated_alias(
    summed_inequality_sample, "summed_inequality_sampler"
)
