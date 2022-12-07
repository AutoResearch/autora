from typing import Iterable, Literal

import numpy as np
from sklearn.metrics import DistanceMetric

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

def summed_inequality_sampler(
    X: np.ndarray, X_ref: np.ndarray, n: int = 1, equality_distance: float = 0,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    This inequality sampler chooses from the pool of IV conditions according to their
    inequality with respect to a reference pool X_ref. Two IVs are considered equal if their
    distance is less then the equality_distance. The IVs chosen first are feed back into X_ref
    and are included in the summed equality calculation.

    Args:
        X: pool of IV conditions to evaluate dissimilarity
        X_ref: reference pool of IV conditions
        n: number of samples to select
        equality_distance: the distance to decide if two data points are equal.
        metric: dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See `sklearn.metrics.DistanceMetric` for more details.

    Returns:
        Sampled pool
    """

    if isinstance(X, Iterable):
        X = np.array(list(X))

    if isinstance(X_ref, Iterable):
        X_ref = np.array(list(X_ref))

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)

    if X.shape[1] != X_ref.shape[1]:
        raise ValueError(
            f"X and X_ref must have the same number of columns.\n"
            f"X has {X.shape[1]} columns, while X_ref has {X_ref.shape[1]} columns."
        )

    if X.shape[0] < n:
        raise ValueError(
            f"X must have at least {n} rows matching the number of requested samples."
        )

    dist = DistanceMetric.get_metric(metric)

    # create a list to store the n X-values with that highest inequality scores
    X_res = []
    # choose the canditate with the highest inequality score n-times
    for _ in range(n):
        summed_equalities = []
        # loop over all IV values
        for row in X:

            # calculate the distances between the current row in matrix1 and all other rows in matrix2
            summed_equality = 0
            for X_ref_row in X_ref:
                distance = dist.pairwise([row, X_ref_row])[0, 1]
                summed_equality += distance > equality_distance

            # store the summed distance for the current row
            summed_equalities.append(summed_equality)



        # sort the rows in matrix1 by their summed distances
        X = X[np.argsort(summed_equalities)[::-1]]
        # append the first value of the sorted list to the result
        X_res.append(X[0])
        # add the chosen value to X_ref
        X_ref = np.append(X_ref, [X[0]], axis=0)
        # remove the chosen value from X
        X = X[1:]

    return np.array(X_res[:n])
