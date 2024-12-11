import itertools
import warnings
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from autora.utils.deprecation import deprecated_alias


def score_sample_custom_distance(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    distance_fct: Callable = lambda x, y: (x - y) ** 2,
    aggregate_fct: Callable = lambda x: np.sum(x, axis=0),
    num_samples: Optional[int] = None,
):
    """
    An experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions. The disagreement
    measurement is customizable.


    Args:
        conditions: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        distance_fct: distance function to use on the predictions
        aggregate_fct: aggregate function to use on the pairwise distances of the models
        num_samples: number of samples to select

    Returns:
        Sampled pool with score


    Examples:
        We can use this without passing in a distance function (squared distance as default) ...
        >>> class IdentityModel:
        ...     def predict(self, X):
        ...         return X
        >>> class SquareModel:
        ...     def predict(self, X):
        ...         return X**2
        >>> id_model = IdentityModel()
        >>> sq_model = SquareModel()
        >>> _conditions = np.array([1, 2, 3])
        >>> id_model.predict(_conditions)
        array([1, 2, 3])
        >>> sq_model.predict(_conditions)
        array([1, 4, 9])
        >>> score_sample_custom_distance(_conditions, [id_model, sq_model])
           0  score
        2  3     36
        1  2      4
        0  1      0

        ... we can use our own distance function (for example binary 1 and 0 for different or equal)
        >>> score_sample_custom_distance(_conditions, [id_model, sq_model], lambda x,y : x != y)
           0  score
        1  2      1
        2  3      1
        0  1      0

        ... this is mostly usefull if the predict function of the model doesn't return a
        standard one-dimensional array:
        >>> _conditions = np.array([[0, 1], [1, 0], [1, 1], [.5, .5]])
        >>> id_model.predict(_conditions)
        array([[0. , 1. ],
               [1. , 0. ],
               [1. , 1. ],
               [0.5, 0.5]])
        >>> sq_model.predict(_conditions)
        array([[0.  , 1.  ],
               [1.  , 0.  ],
               [1.  , 1.  ],
               [0.25, 0.25]])

        >>> def distance(x, y):
        ...     return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

        >>> score_sample_custom_distance(_conditions, [id_model, sq_model], distance)
             0    1     score
        3  0.5  0.5  0.353553
        0  0.0  1.0  0.000000
        1  1.0  0.0  0.000000
        2  1.0  1.0  0.000000
    """
    disagreements = []
    for model_a, model_b in itertools.combinations(models, 2):
        if hasattr(model_a, "predict_proba") and hasattr(model_b, "predict_proba"):
            model_a_predict = model_a.predict_proba
            model_b_predict = model_b.predict_proba
        else:
            model_a_predict = model_a.predict
            model_b_predict = model_b.predict
        y_A = model_a_predict(conditions)
        y_B = model_b_predict(conditions)
        disagreements.append([distance_fct(y_a, y_b) for y_a, y_b in zip(y_A, y_B)])
    score = aggregate_fct(disagreements)

    conditions_new = pd.DataFrame(conditions)
    conditions_new["score"] = np.array(score).tolist()
    conditions_new = conditions_new.sort_values(by="score", ascending=False)
    if num_samples is None:
        return conditions_new
    else:
        return conditions_new.head(num_samples)


def sample_custom_distance(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    distance_fct: Callable = lambda x, y: (x - y) ** 2,
    aggregate_fct: Callable = lambda x: np.sum(x, axis=0),
    num_samples: Optional[int] = 1,
):
    """
    An experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions. The disagreement
    measurement is customizable.

    Args:
        conditions: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        distance_fct: distance function to use on the predictions
        aggregate_fct: aggregate function to use on the pairwise distances of the models
        num_samples: number of samples to select

    Returns: Sampled pool
    """

    selected_conditions = score_sample_custom_distance(
        conditions, models, distance_fct, aggregate_fct, num_samples
    )
    selected_conditions.drop(columns=["score"], inplace=True)
    return selected_conditions


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

    if (
        isinstance(conditions, Iterable)
        and not isinstance(conditions, pd.DataFrame)
        and not isinstance(conditions, list)
    ):
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
                    raise ValueError(
                        "X_predict must be a list of numpy arrays if it is a list."
                    )
                else:
                    disagreement_part = compute_disagreement(
                        model_a, model_b, element, predict_proba
                    )
                    disagreement_part_list.append(disagreement_part)
            disagreement = np.sum(disagreement_part_list, axis=1)
        else:
            disagreement = compute_disagreement(
                model_a, model_b, X_predict, predict_proba
            )
        model_disagreement.append(disagreement)

    assert len(model_disagreement) >= 1, "No disagreements to compare."

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    if isinstance(condition_pool_copy, pd.DataFrame):
        conditions = pd.DataFrame(conditions, columns=condition_pool_copy.columns)
    elif isinstance(condition_pool_copy, list):
        conditions = pd.DataFrame({"X": conditions})
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
        warnings.warn(
            "Found nan or inf values in model predictions, "
            "setting disagreement there to 0"
        )
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
model_disagreement_sampler_custom_distance = sample_custom_distance
model_disagreement_score_sample_custom_distance = score_sample_custom_distance
