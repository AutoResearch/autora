from math import log

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from autora.experimentalist.sampler import uncertainty_sampler


@pytest.fixture
def synthetic_lr_model():
    """
    Creates logistic regression classifier for 3 classes based on synthetic data.
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
            [0.4, 0.3, 0.2],
            [0.4, 0.4, 0.2],
        ]
    )
    return data


def test_uncertainty_least_confident(synthetic_lr_model, data_to_test):
    # Import model and data
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with least confident measure
    samples = uncertainty_sampler(X, model, 5, measure="least_confident")

    # Manual Calculation - Uses slightly different method than the function by using pandas Series
    # and index instead of pure numpy
    mat_prob = model.predict_proba(X)
    a_uncert = 1 - mat_prob.max(axis=1)
    s_uncert = pd.Series(a_uncert).sort_values(ascending=False)
    idx = s_uncert.index.values[0:5]
    manual_samples = X[idx]

    assert np.array_equal(samples, manual_samples)


def test_uncertainty_margin(synthetic_lr_model, data_to_test):
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with margin measure
    samples = uncertainty_sampler(X, model, 5, measure="margin")

    # Manual Calculation
    mat_prob = model.predict_proba(X)
    l_margins = []
    for l_prob in mat_prob:
        l_sort = np.sort(l_prob)
        l_margins.append(l_sort[-1] - l_sort[-2])
    s_margins = pd.Series(l_margins).sort_values(ascending=True)
    idx = s_margins.index.values[0:5]
    manual_samples = X[idx]

    assert np.array_equal(samples, manual_samples)


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
    Test data should yield different results. Condition [0.4, 0.3, 0.2] should have the greatest
    entropy but less margin than conditions [0.4, 0.4, 0.2] and [0.5, 0.5, 0. ].
    """
    model = synthetic_lr_model
    X = data_to_test

    # Run uncertainty sampler with entropy and margin measures to compare
    samples_entropy = uncertainty_sampler(X, model, 5, measure="entropy")
    samples_margin = uncertainty_sampler(X, model, 5, measure="margin")

    assert not np.array_equal(samples_entropy, samples_margin)
