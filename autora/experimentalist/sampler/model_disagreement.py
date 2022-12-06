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
    for idx_A, model_A in enumerate(models):
        for idx_B, model_B in enumerate(models):

            # don't compare the model with itself
            if idx_A <= idx_B:
                continue

            # determine the prediction method
            model_A_predict = getattr(model_A, "predict_proba", None)
            if callable(model_A_predict) is False:
                model_A_predict = getattr(model_A, "predict", None)
                model_B_predict = getattr(model_B, "predict", None)
            else:
                model_B_predict = getattr(model_B, "predict_proba", None)
                if callable(model_B_predict) is False:
                    raise Exception("Models must have the same prediction method.")

            if model_A_predict is None or model_B_predict is None:
                raise Exception("Model must have `predict` or `predict_proba` method.")

            # get predictions from both models
            Y_A = model_A_predict(X_predict)
            Y_B = model_B_predict(X_predict)

            if Y_A.shape != Y_B.shape:
                raise Exception("Models must have same output shape.")

            # determine the disagreement between the two models in terms of mean-squared error
            if len(Y_A.shape) == 1:
                disagreement = (Y_A - Y_B) ** 2
            else:
                disagreement = np.mean((Y_A - Y_B) ** 2, axis=1)

            model_disagreement.append(disagreement)

    if len(model_disagreement) == 0:
        raise Exception("No models to compare.")

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    # sort the summed disagreements
    idx = (-summed_disagreement).argsort()[:num_samples]

    return X[idx]
