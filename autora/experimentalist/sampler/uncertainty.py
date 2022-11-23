from typing import Iterable

import numpy as np
from alipy.query_strategy.query_labels import QueryInstanceUncertainty


def uncertainty_sampler(X, model, n, measure="least_confident"):
    """

    Args:
        X: pool of IV conditions to evaluate uncertainty
        model: Scikitlearn model, must have `predict_proba` method.
        n: number of samples to select
        measure: method to evaluate uncertainty. Options:
        --'least_confident' x* = argmax 1-P(y_hat|x) ,where y_hat = argmax P(yi|x)
        --'margin' x* = argmax P(y_hat1|x) - P(y_hat2|x), where y_hat1 and y_hat2 are the first and
            second most probable class labels under the model, respectively.
        --'entropy' x* = argmax -sum(P(yi|x)logP(yi|x))
        --'distance_to_boundary' Only available in binary classification, x* = argmin |f(x)|,
            your model should have 'decision_function' method which will return a 1d array.

    Returns: Sampled pool

    """
    if isinstance(X, Iterable):
        X = np.array(list(X))
    query = QueryInstanceUncertainty(X=X, measure=measure)
    idx = query.select(
        label_index=[], unlabel_index=np.arange(len(X)), model=model, batch_size=n
    )

    return X[idx]
