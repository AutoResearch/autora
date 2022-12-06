import itertools
from typing import Iterable, List

import numpy as np


def model_disagreement_sampler(X: np.array, models: List, num_samples: int = 1):
    """
    A sampler that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        X: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool
    """

    if isinstance(X, Iterable):
        X = np.array(list(X))

    X_predict = np.array(X)
    if len(X_predict.shape) == 1:
        X_predict = X_predict.reshape(-1, 1)

    model_disagreement = list()

    # collect diagreements for each model apir
    for model_a, model_b in itertools.combinations(models, 2):

        # determine the prediction method
        model_a_predict = getattr(model_a, "predict_proba", None)
        if callable(model_a_predict) is False:
            model_a_predict = getattr(model_a, "predict", None)
            model_b_predict = getattr(model_b, "predict", None)
        else:
            model_b_predict = getattr(model_b, "predict_proba", None)
            if callable(model_b_predict) is False:
                raise Exception("Models must have the same prediction method.")

        if model_a_predict is None or model_b_predict is None:
            raise Exception("Model must have `predict` or `predict_proba` method.")

        # get predictions from both models
        y_a = model_a_predict(X_predict)
        y_b = model_b_predict(X_predict)

        if y_a.shape != y_b.shape:
            raise Exception("Models must have same output shape.")

        # determine the disagreement between the two models in terms of mean-squared error
        if len(y_a.shape) == 1:
            disagreement = (y_a - y_b) ** 2
        else:
            disagreement = np.mean((y_a - y_b) ** 2, axis=1)

        model_disagreement.append(disagreement)

    if len(model_disagreement) == 0:
        raise Exception("No models to compare.")

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    # sort the summed disagreements
    idx = (-summed_disagreement).argsort()[:num_samples]

    return X[idx]
