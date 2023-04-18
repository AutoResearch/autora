import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.sampler.falsification import (
    falsification_sampler,
    get_scored_samples_from_model_prediction,
)
from autora.variable import DV, IV, ValueType, VariableCollection
from tests.test_poppernet_pooler import get_sin_data, get_xor_data

x_min_regression = 0
x_max_regression = 6


@pytest.fixture
def synthetic_logr_model():
    """
    Creates logistic regression classifier for 3 classes based on synthetic data_closed_loop.
    """
    X, y = get_xor_data()
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def synthetic_linr_model():
    """
    Creates linear regression based on synthetic data_closed_loop.
    """
    x, y = get_sin_data()
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


@pytest.fixture
def classification_data_to_test():
    data = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ]
    )
    return data


@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    torch.manual_seed(180)
    return


@pytest.fixture
def get_square_data():
    X = np.linspace(x_min_regression, x_max_regression, 100)
    Y = np.square(X)
    return X, Y


@pytest.fixture
def regression_data_to_test():
    data = np.linspace(x_min_regression, x_max_regression, 11)
    return data


def test_falsification_classification(
    synthetic_logr_model, classification_data_to_test, seed
):

    # Import model and data_closed_loop
    X_train, Y_train = get_xor_data()
    X = classification_data_to_test
    model = synthetic_logr_model

    # Specify independent variables
    iv1 = IV(
        name="x",
        value_range=(0, 5),
        units="intensity",
        variable_label="stimulus 1",
    )

    # specify dependent variables
    dv1 = DV(
        name="y",
        value_range=(0, 1),
        units="class",
        variable_label="class",
        type=ValueType.CLASS,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv1],
        dependent_variables=[dv1],
    )

    # Run falsification sampler
    falsification_pipeline = Pipeline(
        [("sampler", falsification_sampler)],
        params={
            "sampler": dict(
                X=X,
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                metadata=metadata,
                n=2,
                training_epochs=1000,
                training_lr=1e-3,
            ),
        },
    )

    samples = falsification_pipeline.run()

    print(samples)
    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data_closed_loop used for model training

    assert (samples[0, :] == [1, 1]).all or (samples[1, :] == [1, 1]).all


def test_falsification_regression(synthetic_linr_model, regression_data_to_test, seed):

    # Import model and data_closed_loop
    X_train, Y_train = get_sin_data()
    X = regression_data_to_test
    model = synthetic_linr_model

    # specify meta data_closed_loop

    # Specify independent variables
    iv = IV(
        name="x",
        value_range=(0, 2 * np.pi),
        units="intensity",
        variable_label="stimulus",
    )

    # specify dependent variables
    dv = DV(
        name="y",
        value_range=(-1, 1),
        units="real",
        variable_label="response",
        type=ValueType.REAL,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv],
        dependent_variables=[dv],
    )

    falsification_pipeline = Pipeline(
        [("sampler", falsification_sampler)],
        params={
            "sampler": dict(
                X=X,
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                metadata=metadata,
                n=5,
                training_epochs=1000,
                training_lr=1e-3,
                plot=True,
            ),
        },
    )

    sample = falsification_pipeline.run()

    # the first value should be close to one of the local maxima of the
    # sine function
    assert sample[0] == 0 or sample[0] == 6
    if sample[0] == 0:
        assert (
            sample[1] == 6
            or np.round(sample[1], 2) == 1.8
            or np.round(sample[1], 2 == 4.2)
        )

    assert np.round(sample[2], 2) == 1.8 or np.round(sample[2], 2) == 4.2
    if np.round(sample[2], 2) == 1.8:
        assert np.round(sample[3], 2) == 4.2


def test_falsification_regression_without_model(
    synthetic_linr_model, get_square_data, regression_data_to_test, seed
):
    # obtain the data for training the model
    X_train, Y_train = get_square_data

    # obtain candidate conditions to be evaluated
    X = regression_data_to_test

    # reshape data
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)
    X = X.reshape(-1, 1)

    # fit a linear model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # compute model predictions for trained conditions
    Y_predicted = model.predict(X_train)

    # get scores from falsification sampler
    X_selected, scores = get_scored_samples_from_model_prediction(
        X=X,
        Y_predicted=Y_predicted,
        X_train=X_train,
        Y_train=Y_train,
        training_epochs=1000,
        training_lr=1e-3,
        plot=True,
    )

    # # check if the scores are normalized
    # assert np.round(np.mean(scores), 4) == 0
    # assert np.round(np.std(scores), 4) == 1
    #
    # # check if the scores are properly ordered
    # assert scores[0] > scores[1] > scores[2]
    #
    # # check if the right data points were selected
    # assert X_selected[0, 0] == 0 or X_selected[0, 0] == 6
    # assert X_selected[1, 0] == 0 or X_selected[1, 0] == 6
    # assert X_selected[2, 0] == 3


def test_falsification_reconstruction_without_model(
    synthetic_linr_model, get_square_data, regression_data_to_test, seed
):

    # obtain candidate conditions to be evaluated
    X = regression_data_to_test

    # generate sampled conditions
    X_train = np.linspace(x_min_regression, x_max_regression, 100)

    # generate reconstructed data (this data may be produced by an autoencoder)
    X_reconstructed = X_train + np.sin(X_train)

    # get scores from falsification sampler
    X_selected, scores = get_scored_samples_from_model_prediction(
        X=X,
        Y_predicted=X_reconstructed,
        X_train=X_train,
        Y_train=X_train,
        training_epochs=1000,
        training_lr=1e-3,
        plot=True,
    )

    # check if the scores are normalized
    assert np.round(np.mean(scores), 4) == 0
    assert np.round(np.std(scores), 4) == 1

    # check if the scores are properly ordered
    assert scores[0] > scores[1]

    # check if the data points with the highest predicted error were selected
    assert np.round(X_selected[0, 0], 4) == 1.8 or np.round(X_selected[0, 0], 4) == 4.8
    assert np.round(X_selected[1, 0], 4) == 1.8 or np.round(X_selected[1, 0], 4) == 4.8
