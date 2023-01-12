from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def assumption_sampler(
    X, y, model, n, loss=True, theorist=None, confirmation_bias=False
):
    """
    Assumption Sampler challenges assumptions made by the Theorist.
    It identifies points whose error are most dependent on the assumption made.
    Assumptions take the form of hard-coding, which may be hyperparameters or arbitrarily chosen
    sub-algorithms e.g. loss function
    Because it samples with respect to a Theorist, this sampler cannot be used on the first cycle

    Args:
        X: pool of IV conditions to sample from
        y: experimental results from most recent iteration
        model: Scikit-learn model, must have `predict` method.
        n: number of samples to select
        loss: assumption to test: identify points that are most affected by choice of loss function
        theorist: the Theorist, which employs the theory it has been hard-coded to demonstrate
        confirmation_bias: whether to find evidence to support or oppose the theory

    Returns: Sampled pool

    """

    if isinstance(X, Iterable):
        X = np.array(list(X))
    current = None
    if theorist:
        pass  # add code to extract loss function from theorist object
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
            current_loss = current(
                y_true=y.reshape(1, -1),
                y_pred=y_pred.reshape(1, -1),
                multioutput="raw_values",
            )
            print(current_loss)
            alternative = mae
            alternative_loss = alternative(
                y_true=y.reshape(1, -1),
                y_pred=y_pred.reshape(1, -1),
                multioutput="raw_values",
            )
            loss_delta = alternative_loss - current_loss
            idx = np.flip(loss_delta.argsort()[:n])
    else:
        raise TypeError(
            "Experiment results are required to run the assumption experimentalist"
        )

    return X[idx]
