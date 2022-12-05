from typing import Iterable

import numpy as np


def dissimilarity_sampler(X: np.ndarray, X_base: np.ndarray, n: int = 1):
    """
    This dissimilarity samples re-arranges the pool of IV conditions according to their
    dissimilarity with respect to a reference pool X_base. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in X and X_base.

    TODO: The user may specify different dissimilarity measures.

    Args:
        X: pool of IV conditions to evaluate dissimilarity
        X_base: reference pool of IV conditions
        n: number of samples to select

    Returns:
        Sampled pool
    """

    if isinstance(X, Iterable):
        X = np.array(list(X))

    if isinstance(X_base, Iterable):
        X_base = np.array(list(X_base))

    if X.shape[0] < n:
        raise ValueError(f"X must have at least {n} rows.")

    # create a list to store the summed distances for each row in matrix1
    summed_distances = []

    # loop over each row in matrix1
    for i, row in enumerate(X):
        # calculate the distances between the current row in matrix1 and all other rows in matrix2

        row_distances = []
        for X_base_row in X_base:
            distance = np.sum(np.abs(row - X_base_row))
            row_distances.append(distance)

        # calculate the summed distance for the current row
        summed_distance = np.sum(row_distances)

        # store the summed distance for the current row
        summed_distances.append(summed_distance)

    # sort the rows in matrix1 by their summed distances
    sorted_X = X[np.argsort(summed_distances)[::-1]]

    return sorted_X[:n]
