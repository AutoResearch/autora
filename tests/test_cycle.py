#!/usr/bin/env python
import numpy as np
import pytest  # noqa: 401

from autora.cycle import (  # noqa: 401
    DataSetCollection,
    run,
    start_experiment_runner,
    start_experimentalist,
    start_theorist,
)
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
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        search_space=None,
        data=DataSetCollection([]),
        cycle_count=0,
        theory=None,
        independent_variable_values=x1,
        max_cycle_count=10,
    )

    # Run from experiment runner
    experiment_runner_results_run = run(
        starting_point="experiment_runner", **parameters
    )
    print(experiment_runner_results_run)

    # Run starting from theorist
    theorist_results_run = run(starting_point="theorist", **parameters)
    print(theorist_results_run)

    # Run starting from experimentalist
    experimentalist_results_run = run(starting_point="experimentalist", **parameters)
    print(experimentalist_results_run)


if __name__ == "__main__":
    test_cycle()
