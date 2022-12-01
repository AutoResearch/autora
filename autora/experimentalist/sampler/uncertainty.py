from typing import Iterable

import numpy as np
from scipy.stats import entropy


def uncertainty_sampler(X, model, n, measure="least_confident"):
    """

    Args:
        X: pool of IV conditions to evaluate uncertainty
        model: Scikit-learn model, must have `predict_proba` method.
        n: number of samples to select
        measure: method to evaluate uncertainty. Options:

            - `'least_confident'`: $x* = \\operatorname{argmax} \\left( 1-P(\\hat{y}|x) \\right)$,
              where $\\hat{y} = \\operatorname{argmax} P(y_i|x)$
            - `'margin'`:
              $x* = \\operatorname{argmax} \\left( P(\\hat{y}_1|x) - P(\\hat{y}_2|x) \\right)$,
              where $\\hat{y}_1$ and $\\hat{y}_2$ are the first and second most probable
              class labels under the model, respectively.
            - `'entropy'`:
              $x* = \\operatorname{argmax} \\left( - \\sum P(y_i|x)
              \\operatorname{log} P(y_i|x) \\right)$

    Returns: Sampled pool

    """

    if isinstance(X, Iterable):
        X = np.array(list(X))

    a_prob = model.predict_proba(X)

    if measure == "least_confident":
        # Calculate uncertainty of max probability class
        a_uncertainty = 1 - a_prob.max(axis=1)
        # Get index of largest uncertainties
        idx = np.flip(a_uncertainty.argsort()[-n:])

    elif measure == "margin":
        # Sort values by row descending
        a_part = np.partition(-a_prob, 1, axis=1)
        # Calculate difference between 2 largest probabilities
        a_margin = -a_part[:, 0] + a_part[:, 1]
        # Determine index of smallest margins
        idx = a_margin.argsort()[:n]

    elif measure == "entropy":
        # Calculate entropy
        a_entropy = entropy(a_prob.T)
        # Get index of largest entropies
        idx = np.flip(a_entropy.argsort()[-n:])

    else:
        raise ValueError(
            f"Unsupported uncertainty measure: '{measure}'\n"
            f"Only 'least_confident', 'margin', or 'entropy' is supported."
        )

    return X[idx]
