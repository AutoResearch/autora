#!/usr/bin/env python
import copy
from functools import partial

import numpy as np
import pytest  # noqa: 401
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from autora.cycle import AERModule, DataSetCollection, TheoryCollection, run
from autora.skl.darts import DARTSRegressor
from autora.variable import Variable, VariableCollection


def test_cycle():
    """
    This test prototypes the design and calling of the closed cycle system starting from
    different modules of the cycle (>> Experiment Runner > Theorist > Experimentalist >>).
    This prototype uses simplistic representations of the theorist, experimentalist, and experiment
    runner.

    theorist: Generates a constant theory of  y = x + 1, independent of inputs.
    experimentalist: Generates new independent variables (x') to experiment from a uniform random
                     selection of values from a defined range. Independent of inputs.
    experiment_runner: Simulates data gained from an experiment that perfectly matches the theory
                       y = x' + 1

    This test was first conceptualized from the exercise of starting from the experiment runner.
    Seed x' (Independent variable) values are inputs to experiment runner to start the cycle and
    create syntheticexperimental data.

    Notes on seed requirements:
    1. Experiment Runner Start
        a. Independent variable values
    2. Theorist Start
        a. Paired independent variable values and dependent variable values
    3. Experimentalist Start
        a. Theory
        b. Paired independent variable values and dependent variable values

    """

    # Define basic versions of the modules
    def dummy_theorist(data, metadata, search_space):
        def theory(x):
            return x + 1

        return theory

    def dummy_experimentalist(data, metadata: VariableCollection, theory):
        low = metadata.independent_variables[0].min
        high = metadata.independent_variables[0].max
        x_prime = np.random.uniform(low, high, 10)
        return x_prime

    def dummy_experiment_runner(x_prime):
        return x_prime + 1

    #  Define parameters for run
    x1 = np.linspace(0, 1, 10)  # Seed x' to input into the experiment runner
    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )

    parameters = dict(
        metadata=metadata,
        search_space=None,
        data=DataSetCollection([]),
        theories=TheoryCollection([lambda x: x + 2]),
        independent_variable_values=x1,
        max_cycle_count=10,
        cycle_count=0,
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
    )

    # Run from experiment runner
    experiment_runner_results_run = run(
        first_state=AERModule.EXPERIMENT_RUNNER, **parameters
    )
    print(experiment_runner_results_run)

    # Run starting from theorist
    theorist_results_run = run(first_state=AERModule.THEORIST, **parameters)
    print(theorist_results_run)

    # Run starting from experimentalist
    experimentalist_results_run = run(
        first_state=AERModule.EXPERIMENTALIST, **parameters
    )
    print(experimentalist_results_run)


def test_sklearn_theorist():
    """
    This test prototypes the design and calling of the closed cycle system using a SciKit-Learn
    regressor as the theorist.

    theorist: Generates a constant theory of  y = x + 1, independent of inputs.
    experimentalist: Generates new independent variables (x') to experiment from a uniform random
                     selection of values from a defined range. Independent of inputs.
    experiment_runner: Simulates data gained from an experiment that perfectly matches the theory
                       y = x' + 1

    This test was first conceptualized from the exercise of starting from the experiment runner.
    Seed x' (Independent variable) values are inputs to experiment runner to start the cycle and
    create syntheticexperimental data.

    Notes on seed requirements:
    1. Experiment Runner Start
        a. Independent variable values
    2. Theorist Start
        a. Paired independent variable values and dependent variable values
    3. Experimentalist Start
        a. Theory
        b. Paired independent variable values and dependent variable values

    """

    def scikit_learn_regressor_theorist_wrapper(theorist: SklearnBaseEstimator):
        def theorist_(data, metadata, search_space):
            theorist_ = copy.copy(theorist)
            X = np.vstack([d.x.reshape(-1, 1) for d in data.datasets if d is not None])
            y = np.vstack([d.y.reshape(-1, 1) for d in data.datasets if d is not None])
            theorist_.fit(X, y)
            theory = theorist_.predict
            return theory

        return theorist_

    def make_random_experimentalist(n_observations_per_experiment=10):
        def dummy_experimentalist(data, metadata: VariableCollection, theory):
            low = metadata.independent_variables[0].min
            high = metadata.independent_variables[0].max
            x_prime = np.random.uniform(low, high, n_observations_per_experiment)
            return x_prime

        return dummy_experimentalist

    def x_plus_one(x):
        return x + 1

    random_number_generator = np.random.default_rng(seed=42)
    noise_generator = partial(random_number_generator.normal, loc=0.0, scale=0.5)

    def instantiated_noise_source(x):
        x_noisy = x + noise_generator(size=x.shape)
        return x_noisy

    def make_synthetic_experiment_runner(ground_truth_theory, noise_source):
        def synthetic_experiment_runner(x_prime):
            measurement = noise_source(ground_truth_theory(x_prime))
            return measurement

        return synthetic_experiment_runner

    #  Define parameters for run
    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )

    # Run starting from experimentalist
    experimentalist_results_run = run(
        first_state=AERModule.EXPERIMENTALIST,
        theorist=scikit_learn_regressor_theorist_wrapper(
            DARTSRegressor(
                num_graph_nodes=2,
                max_epochs=1000,
                primitives=(
                    "none",
                    "add",
                    "subtract",
                    "linear",
                    "relu",
                ),
            )
        ),
        experimentalist=make_random_experimentalist(n_observations_per_experiment=50),
        experiment_runner=make_synthetic_experiment_runner(
            x_plus_one, noise_source=instantiated_noise_source
        ),
        metadata=metadata,
        search_space=None,
        data=DataSetCollection([None]),
        theories=TheoryCollection([None]),
        independent_variable_values=None,
        max_cycle_count=10,
        cycle_count=0,
    )
    print(experimentalist_results_run)
    x_test = np.linspace(
        metadata.independent_variables[0].min,
        metadata.independent_variables[0].max,
        100,
    ).reshape(-1, 1)
    fig = plt.figure()
    for data in experimentalist_results_run.data.datasets:
        if data is not None:
            plt.scatter(data.x, data.y)
    plt.plot(x_test, experimentalist_results_run.theories[-1](x_test), c="orange")
    fig.show()
