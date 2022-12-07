from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def assumption_sampler(X, model, n, y=None, loss=True, theory=None):
    """

    Args:
        X: pool of IV conditions to sample from
        model: Scikit-learn model, must have `predict_proba` method.
        n: number of samples to select

    Returns: Sampled pool

    """

    if isinstance(X, Iterable):
        X = np.array(list(X))

    current = None
    if theory:
        pass  # add code to extract loss function used from theory object
    idx = range(len(X))

    if y is not None:
        if loss:
            if current is None:
                current = mse
                print(
                    Warning(
                        "Knowledge of Theorist Loss Function needed. MSE has been assumed."
                    )
                )
            y_pred = model.predict(X)
            current_loss = current(y_true=y, y_pred=y_pred, multioutput="raw_values")
            alternative = mae
            alternative_loss = alternative(
                y_true=y, y_pred=y_pred, multioutput="raw_values"
            )
            loss_delta = alternative_loss - current_loss
            idx = np.flip(loss_delta.argsort()[-n:])
    else:
        raise TypeError(
            "Experiment results are required to run the assumption experimentalist"
        )

    return X[idx]
