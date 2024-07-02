"""
Novelty Experimentalist
"""
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler

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
    num_samples: Optional[int] = None,
    metric: AllowedMetrics = "euclidean",
    integration: str = "min",
):
    """
    This novelty experimentalist re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference
    conditions.
    If no number of samples are specified, all samples will be ordered and returned from the pool.

    Args:
        conditions: pool of experimental conditions to evaluate dissimilarity
        reference_conditions: reference pool of experimental conditions
        num_samples: number of samples to select from the pool of experimental conditions
        (the default is to select all)
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.

    Returns:
        Sampled pool of conditions
    """

    condition_pool_copy = conditions.copy()

    new_conditions = novelty_score_sample(
        conditions, reference_conditions, num_samples, metric, integration
    )
    new_conditions.drop("score", axis=1, inplace=True)

    if isinstance(condition_pool_copy, pd.DataFrame):
        new_conditions = pd.DataFrame(new_conditions, columns=condition_pool_copy.columns)

    return new_conditions


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: Optional[int] = None,
    metric: AllowedMetrics = "euclidean",
    integration: str = "sum",
) -> pd.DataFrame:
    """
    This dissimilarity samples re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference
    conditions.
    If no number of samples are specified, all samples will be ordered and returned from the pool.

    Args:
        conditions: pool of experimental conditions to evaluate dissimilarity
        reference_conditions: reference pool of experimental conditions
        num_samples: number of samples to select from the pool of experimental conditions
        (the default is to select all)
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.
        integration: Distance integration method used to compute the overall dissimilarity score
        for a given data point. Options: 'sum', 'prod', 'mean', 'min', 'max'.

    Returns:
        Sampled pool of conditions and dissimilarity scores
    """
    conditions = pd.DataFrame(conditions)
    reference_conditions = pd.DataFrame(reference_conditions)

    dist = DistanceMetric.get_metric(metric)
    distances = dist.pairwise(reference_conditions, conditions)

    if integration == "sum":
        integrated_distance = np.sum(distances, axis=0)
    elif integration == "mean":
        integrated_distance = np.mean(distances, axis=0)
    elif integration == "max":
        integrated_distance = np.max(distances, axis=0)
    elif integration == "min":
        integrated_distance = np.min(distances, axis=0)
    elif integration == "prod":
        integrated_distance = np.prod(distances, axis=0)
    else:
        raise ValueError(f"Integration method {integration} not supported.")

    # normalize the distances
    scaler = StandardScaler()
    score = scaler.fit_transform(integrated_distance.reshape(-1, 1)).flatten()

    # order rows in Y from highest to lowest
    conditions["score"] = score
    conditions = conditions.sort_values(by="score", ascending=False)

    if num_samples is not None:
        return conditions[:num_samples]
    else:
        return conditions


novelty_sample = sample
novelty_sample.__doc__ = """Alias for sample"""
novelty_score_sample = score_sample
novelty_score_sample.__doc__ = """Alias for score_sample"""
novelty_sampler = deprecated_alias(novelty_sample, "novelty_sampler")
novelty_score_sampler = deprecated_alias(novelty_score_sample, "novelty_score_sampler")
