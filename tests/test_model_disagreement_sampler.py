import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from autora.experimentalist.sampler.model_disagreement import model_disagreement_sampler


def get_classification_data(n: int = 100):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    # cross product of x1 and x2
    X = np.array([(x1[i], x2[j]) for i in range(len(x1)) for j in range(len(x2))])

    # create a vector of 0s and 1s which is 0 whenever x1 < 0.5 and x2 < 0.5 and 1 otherwise
    y_A = np.zeros(n * n)
    y_B = np.zeros(n * n)
    y_A[(X[:, 0] >= 0.5) | (X[:, 1] >= 0.5)] = 1
    y_B[(X[:, 0] >= 0.5)] = 1

    return X, y_A, y_B


def get_polynomial_data(n: int = 100):
    x = np.linspace(-1, 1, 100)
    y = x**2
    return x, y


@pytest.fixture
def synthetic_lr_models():
    """
    Creates two logistic regression classifier for 2 classes based on synthetic data_closed_loop.
    Each classifier is trained on a different data_closed_loop set and thus should yield different predictions.
    """
    X, y_A, y_B = get_classification_data()
    model_A = LogisticRegression()
    model_B = LogisticRegression()
    model_A.fit(X, y_A)
    model_B.fit(X, y_B)

    models = [model_A, model_B]
    return models


@pytest.fixture
def synthetic_linr_model():
    """
    Creates linear regression based on synthetic data_closed_loop.
    """
    x, y = get_polynomial_data()
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


@pytest.fixture
def synthetic_poly_model():
    """
    Creates polynomial regression based on synthetic data_closed_loop.
    """
    x, y = get_polynomial_data()

    # define the steps in the pipeline
    steps = [
        (
            "poly",
            PolynomialFeatures(degree=3),
        ),  # transform input data_closed_loop into polynomial features
        ("lr", LinearRegression()),  # fit a linear regression model
    ]
    # create the pipeline
    model = Pipeline(steps)
    model.fit(x.reshape(-1, 1), y)
    return model


@pytest.fixture
def classification_data_to_test(n=10):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)

    # cross product of x1 and x2
    X = np.array([(x1[i], x2[j]) for i in range(len(x1)) for j in range(len(x2))])
    return X


@pytest.fixture
def regression_data_to_test(n=100):
    data = np.linspace(-2, 2, n)
    return data


def test_model_disagreement_classification(
    synthetic_lr_models, classification_data_to_test
):

    num_requested_samples = 10

    # Import model and data_closed_loop
    X = classification_data_to_test
    models = synthetic_lr_models

    # Run model disagreement sampler
    samples = model_disagreement_sampler(X, models, num_requested_samples)

    assert samples.shape[0] == num_requested_samples
    assert samples[0, 0] < 0.25 and samples[0, 1] > 0.75
    assert samples[1, 0] < 0.25 and samples[1, 1] > 0.75


def test_model_disagreement_regression(
    synthetic_linr_model, synthetic_poly_model, regression_data_to_test
):

    num_requested_samples = 2

    # Import model and data_closed_loop
    X = regression_data_to_test
    model_A = synthetic_linr_model
    model_B = synthetic_poly_model
    models = [model_A, model_B]

    # Run model disagreement sampler
    samples = model_disagreement_sampler(X, models, num_requested_samples)

    assert len(samples) == num_requested_samples
    assert samples[0] == 2.0 or samples[0] == -2.0
    assert samples[1] == 2.0 or samples[1] == -2.0
