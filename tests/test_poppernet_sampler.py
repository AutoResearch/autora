import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.sampler.poppernet import (
    nearest_values_sampler,
    poppernet_pooler,
)
from autora.variable import DV, IV, ValueType, VariableCollection


def get_xor_data(n: int = 3):
    X = ([[1, 0]] * n) + ([[0, 1]] * n) + ([[0, 0]] * n) + ([[1, 1]])
    y = ([0] * n) + ([0] * n) + ([1] * n) + ([1])
    return X, y


def get_sin_data(n: int = 100):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    return x, y


@pytest.fixture
def synthetic_logr_model():
    """
    Creates logistic regression classifier for 3 classes based on synthetic data.
    """
    X, y = get_xor_data()
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def synthetic_linr_model():
    """
    Creates linear regression based on synthetic data.
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
def regression_data_to_test():
    data = [-10, 0, 1.5, 3, 4.5, 6, 10]
    return data


def test_poppernet_classification(synthetic_logr_model, classification_data_to_test):

    # Import model and data
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

    # Run popper net sampler
    poppernet_pipeline = Pipeline(
        [("pool", poppernet_pooler), ("sampler", nearest_values_sampler)],
        params={
            "pool": dict(
                model=model,
                x_train=X_train,
                y_train=Y_train,
                metadata=metadata,
                num_samples=2,
                training_epochs=1000,
                optimization_epochs=1000,
                training_lr=1e-3,
                optimization_lr=1e-3,
                mse_scale=1,
                limit_offset=10**-10,
                limit_repulsion=0,
            ),
            "sampler": {"allowed_values": X},
        },
    )

    samples = poppernet_pipeline.run()

    print(samples)
    # Check that at least one of the resulting samples is the one that is
    # underrepresented in the data used for model training

    assert (samples[0, :] == [1, 1]).all or (samples[1, :] == [1, 1]).all


def test_poppernet_regression(synthetic_linr_model, regression_data_to_test):

    # Import model and data
    X_train, Y_train = get_sin_data()
    X = regression_data_to_test
    model = synthetic_linr_model

    # specify meta data

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

    poppernet_pipeline = Pipeline(
        [("pool", poppernet_pooler), ("sampler", nearest_values_sampler)],
        params={
            "pool": dict(
                model=model,
                x_train=X_train,
                y_train=Y_train,
                metadata=metadata,
                num_samples=5,
                training_epochs=1000,
                optimization_epochs=1000,
                training_lr=1e-3,
                optimization_lr=1e-3,
                mse_scale=1,
                limit_offset=10**-10,
                limit_repulsion=0,
            ),
            "sampler": {"allowed_values": X},
        },
    )

    sample = poppernet_pipeline.run()

    # the first value should be close to one of the local maxima of the
    # sine function
    assert sample[0] == 1.5 or sample[0] == 4.5
