import pickle
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import autora.utils.YAMLSerializer as YAMLSerializer
from autora.cycle.simple import SimpleCycle, SimpleCycleData
from autora.experimentalist.pipeline import make_pipeline
from autora.variable import Variable, VariableCollection


@pytest.fixture
def logistic_regression_dataset():
    def ground_truth(x):
        return x + 1

    metadata_0 = VariableCollection(
        independent_variables=[Variable(name="x1", allowed_values=range(11))],
        dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    )
    example_experimentalist = make_pipeline(
        [metadata_0.independent_variables[0].allowed_values]
    )

    def get_example_synthetic_experiment_runner():
        rng = np.random.default_rng(seed=180)

        def runner(x):
            return ground_truth(x) + rng.normal(0, 0.1, x.shape)

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()
    example_synthetic_experiment_runner(np.ndarray([1]))
    example_theorist = LinearRegression()
    cycle = SimpleCycle(
        metadata=metadata_0,
        theorist=example_theorist,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
        monitor=lambda data: print(f"Generated {len(data.theories)} theories"),
    )
    cycle.run(num_cycles=3)

    return cycle.data


@pytest.fixture
def darts_dataset():
    return


def test_pickle_save_load_logistic_regression(
    logistic_regression_dataset: SimpleCycleData,
):
    run_save_load_test(
        pickle, logistic_regression_dataset, assert_equality_logistic_regression_dataset
    )


def test_joblib_save_load_logistic_regression(
    logistic_regression_dataset: SimpleCycleData,
):
    run_save_load_test(
        joblib, logistic_regression_dataset, assert_equality_logistic_regression_dataset
    )


def test_yaml_save_load_logistic_regression(
    logistic_regression_dataset: SimpleCycleData,
):
    run_save_load_test(
        YAMLSerializer,
        logistic_regression_dataset,
        assert_equality_logistic_regression_dataset,
        filetype="w+",
    )


def run_save_load_test(serializer, data, comparator, filetype="w+b"):
    with tempfile.NamedTemporaryFile(filetype) as file:
        serializer.dump(data, file)

        file.seek(0)
        reloaded_data = serializer.load(file)

        comparator(data, reloaded_data)


def assert_equality_logistic_regression_dataset(a, b):
    assert a.metadata == b.metadata
    for a_theory, b_theory in zip(a.theories, b.theories):
        assert a_theory.coef_ == b_theory.coef_
    for a_conditions, b_conditions in zip(a.conditions, b.conditions):
        assert np.array_equal(a_conditions, b_conditions, equal_nan=True)
    for a_observations, b_observations in zip(b.observations, a.observations):
        assert np.array_equal(a_observations, b_observations, equal_nan=True)
