import itertools
import warnings
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from autora.utils.deprecation import deprecated_alias


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    num_samples: Optional[int] = None,
):
    """
    A experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        conditions: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool

    Examples:
        If a model is undefined at a certain condition, the disagreement on that point is set to 0:
        >>> class ModelUndefined:
        ...     def predict(self, X):
        ...         return np.log(X)
        >>> class ModelDefinined:
        ...     def predict(self, X):
        ...         return X
        >>> modelUndefined = ModelUndefined()
        >>> modelDefined = ModelDefinined()
        >>> conditions_defined = np.array([1, 2, 3])
        >>> score_sample(conditions_defined, [modelUndefined, modelDefined], 3)
           0     score
        2  3  1.364948
        1  2 -0.362023
        0  1 -1.002924

        >>> conditions_undefined = np.array([-1, 0, 1, 2, 3])
        >>> score_sample(conditions_undefined, [modelUndefined, modelDefined], 5)
           0     score
        4  3  1.752985
        3  2  0.330542
        2  1 -0.197345
        0 -1 -0.943091
        1  0 -0.943091
    """

    if isinstance(conditions, Iterable) and not isinstance(conditions, pd.DataFrame) and not isinstance(conditions, list):
        conditions = np.array(list(conditions))

    condition_pool_copy = conditions.copy()

    if isinstance(conditions, list):
        X_predict = conditions
    else:
        conditions = np.array(conditions)
        X_predict = np.array(conditions)
        if len(X_predict.shape) == 1:
            X_predict = X_predict.reshape(-1, 1)

    model_disagreement = list()

    # collect diagreements for each model pair
    for model_a, model_b in itertools.combinations(models, 2):

        # determine the prediction method
        predict_proba = False
        if hasattr(model_a, "predict_proba") and hasattr(model_b, "predict_proba"):
            predict_proba = True
        elif hasattr(model_a, "predict") and hasattr(model_b, "predict"):
            predict_proba = False
        else:
            raise AttributeError(
                "Models must both have `predict_proba` or `predict` method."
            )

        if isinstance(X_predict, list):
            disagreement_part_list = list()
            for element in X_predict:
                if not isinstance(element, np.ndarray):
                    raise ValueError("X_predict must be a list of numpy arrays if it is a list.")
                else:
                    disagreement_part = compute_disagreement(model_a, model_b, element, predict_proba)
                    disagreement_part_list.append(disagreement_part)
            disagreement = np.sum(disagreement_part_list, axis=1)
        else:
            disagreement = compute_disagreement(model_a, model_b, X_predict, predict_proba)

        model_disagreement.append(disagreement)

    assert len(model_disagreement) >= 1, "No disagreements to compare."

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    if isinstance(condition_pool_copy, pd.DataFrame):
        conditions = pd.DataFrame(conditions, columns=condition_pool_copy.columns)
    elif isinstance(condition_pool_copy, list):
        conditions = pd.DataFrame({'X': conditions})
    else:
        conditions = pd.DataFrame(conditions)

    # normalize the distances
    scaler = StandardScaler()
    score = scaler.fit_transform(summed_disagreement.reshape(-1, 1)).flatten()

    # order rows in Y from highest to lowest
    conditions["score"] = score
    conditions = conditions.sort_values(by="score", ascending=False)

    if num_samples is None:
        return conditions
    else:
        return conditions.head(num_samples)

def compute_disagreement(model_a, model_b, X_predict, predict_proba):
    # get predictions from both models
    if predict_proba:
        y_a = model_a.predict_proba(X_predict)
        y_b = model_b.predict_proba(X_predict)
    else:
        y_a = model_a.predict(X_predict)
        y_b = model_b.predict(X_predict)

    assert y_a.shape == y_b.shape, "Models must have same output shape."

    # determine the disagreement between the two models in terms of mean-squared error
    if len(y_a.shape) == 1:
        disagreement = (y_a - y_b) ** 2
    else:
        disagreement = np.mean((y_a - y_b) ** 2, axis=1)

    if np.isinf(disagreement).any() or np.isnan(disagreement).any():
        warnings.warn('Found nan or inf values in model predictions, '
                      'setting disagreement there to 0')
    disagreement[np.isinf(disagreement)] = 0
    disagreement = np.nan_to_num(disagreement)
    return disagreement

def sample(
    conditions: Union[pd.DataFrame, np.ndarray], models: List, num_samples: int = 1
):
    """
    A experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        conditions: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool
    """

    selected_conditions = score_sample(conditions, models, num_samples)
    selected_conditions.drop(columns=["score"], inplace=True)

    return selected_conditions


model_disagreement_sample = sample
model_disagreement_score_sample = score_sample
model_disagreement_sampler = deprecated_alias(
    model_disagreement_sample, "model_disagreement_sampler"
)
