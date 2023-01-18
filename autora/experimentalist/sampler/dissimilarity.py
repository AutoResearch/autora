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


def dissimilarity_sampler(
    X: np.ndarray, X_ref: np.ndarray, n: int = 1, metric: AllowedMetrics = "euclidean",
    inverse: bool = False, integration: str = "sum",
) -> np.ndarray:
    """
    This dissimilarity samples re-arranges the pool of IV conditions according to their
    dissimilarity with respect to a reference pool X_ref. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in X and X_ref.

    Args:
        X: pool of IV conditions to evaluate dissimilarity
        X_ref: reference pool of IV conditions
        n: number of samples to select
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.

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

    X_new = np.zeros((n, X.shape[1]))

    for sample in range(n):

        # normalize all independent variables
        X_full = np.vstack((X, X_ref))
        X_full_norm = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0)
        X_norm = X_full_norm[:X.shape[0]]
        X_ref_norm = X_full_norm[-X_ref.shape[0]:]

        dist = DistanceMetric.get_metric(metric)

        # create a list to store the summed distances for each row in matrix1
        summed_distances = []

        # for each allowed data point
        for row in X_norm:
            # calculate the distances between the current row in matrix1 and all other rows in matrix2
            if integration == "sum":
                integrated_distance = 0
            elif integration == "max":
                integrated_distance = 0
            elif integration == "min":
                integrated_distance = 1e16
            elif integration == "product":
                integrated_distance = 1
            else:
                raise ValueError(f"Integration method {integration} not supported.")

            for X_ref_row in X_ref_norm:

                distance = dist.pairwise([row, X_ref_row])[0, 1]
                if inverse:
                    distance = 1/distance

                if integration == "sum":
                    integrated_distance += distance
                elif integration == "max":
                    integrated_distance = max(integrated_distance, distance)
                elif integration == "min":
                    integrated_distance = min(integrated_distance, distance)
                elif integration == "product":
                    integrated_distance *= distance
                else:
                    raise ValueError(f"Integration method {integration} not supported.")

            # store the summed distance for the current row
            summed_distances.append(integrated_distance)

        # sort the rows in matrix1 by their summed distances
        if inverse:
            sorted_X = X[np.argsort(summed_distances)]
        else:
            sorted_X = X[np.argsort(summed_distances)[::-1]]

        X_add = sorted_X[0]
        X_new[sample] = X_add
        X = np.vstack((X, X_add))

    return sorted_X[:n]
