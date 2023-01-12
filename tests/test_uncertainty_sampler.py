from math import log

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from autora.experimentalist.sampler import uncertainty_sampler


@pytest.fixture
def synthetic_lr_model():
    """
    Creates logistic regression classifier for 3 classes based on synthetic data_closed_loop.
    """
    n = 100
    X = ([[1, 0, 0]] * n) + ([[0, 1, 0]] * n) + ([[0, 0, 1]] * n)
    y = ([0] * n) + ([1] * n) + ([2] * n)
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def data_to_test():
    data = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0.8, 0.2, 0],
            [0.7, 0.3, 0],
            [0.6, 0.4, 0],
            [0.5, 0.5, 0],
            [0.4, 0.3, 0.2],  # doesn't sum to 1
            [0.4, 0.4, 0.2],
        ]
    )
    return data


def test_uncertainty_least_confident(synthetic_lr_model, data_to_test):
    # Import model and data_closed_loop
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with least confident measure
    samples = uncertainty_sampler(X, model, 5, measure="least_confident")

    assert np.array_equal(
        samples,
        np.array(
            [
                # Least confident because 0.4s are equal
                # and 0.2 is different again
                [0.4, 0.4, 0.2],
                [0.5, 0.5, 0],
                [0.4, 0.3, 0.2],
                [0.6, 0.4, 0],
                # Most confident of the least-confident 5 because 0.7 is higher
                # than all the other highest probabilities
                [0.7, 0.3, 0],
            ]
        ),
    )


def test_uncertainty_margin(synthetic_lr_model, data_to_test):
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with margin measure
    samples = uncertainty_sampler(X, model, 5, measure="margin")

    assert np.array_equal(
        samples,
        np.array(
            [
                [
                    0.4,
                    0.4,
                    0.2,
                ],  # For numerical reasons, this comes out first even though ...
                [
                    0.5,
                    0.5,
                    0,
                ],  # ... the margin between class 0 and 1 in *this* case is also zero
                [0.4, 0.3, 0.2],
                [0.6, 0.4, 0],
                [0.7, 0.3, 0],
            ]
        ),
    )


def test_uncertainty_entropy(synthetic_lr_model, data_to_test):
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with margin measure
    samples = uncertainty_sampler(X, model, 5, measure="entropy")

    # Manual Calculation
    mat_prob = model.predict_proba(X)
    l_entropy = []
    for l_prob in mat_prob:
        l_entropy.append(-np.sum([s * log(s) for s in l_prob]))
    s_entropy = pd.Series(l_entropy).sort_values(ascending=False)
    idx = s_entropy.index.values[0:5]
    manual_samples = X[idx]

    assert np.array_equal(samples, manual_samples)


def test_uncertainty_entropy_vs_margin(synthetic_lr_model, data_to_test):
    """
    Test data_closed_loop should yield different results. Condition [0.4, 0.3, 0.2] should have the greatest
    entropy but less margin than conditions [0.4, 0.4, 0.2] and [0.5, 0.5, 0. ].
    """
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with entropy and margin measures to compare
    samples_entropy = uncertainty_sampler(X, model, 1, measure="entropy")
    samples_margin = uncertainty_sampler(X, model, 2, measure="margin")

    assert np.array_equal(samples_entropy, np.array([[0.4, 0.3, 0.2]]))
    assert np.array_equal(samples_margin, np.array([[0.4, 0.4, 0.2], [0.5, 0.5, 0.0]]))
