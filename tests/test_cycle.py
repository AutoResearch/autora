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
    Different starting points of the cycle are invoked by handlers with the prefix `start_<module>`.
    This prototype uses simplistic representations of the theorist, experimentalist, and experiment
    runner.

    theorist: Generates a constant theory of  y = x + 1
    experimentalist: Generates new independent variables (x') to experiment from a uniform random
                     selection of values from a defined range.
    experiment_runner: Represents data gained from an experiment that perfectly matches the theory
                       y = x' + 1

    """

    # Define basic versions of
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
    x1 = np.linspace(0, 1, 100)
    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )
    parameters = dict(
        cycle_count=0,
        max_cycle_count=10,
        experiment_runner=dummy_experiment_runner,
        independent_variable_values=x1,
        data=DataSetCollection([]),
        theorist=dummy_theorist,
        search_space=None,
        metadata=metadata,
        experimentalist=dummy_experimentalist,
        theory=None,
    )

    # Test invoking the cycle at different start points using start handlers
    # Run from experiment runner
    experiment_runner_results = start_experiment_runner(**parameters)
    print(experiment_runner_results)
    # # Run starting from theorist
    theorist_results = start_theorist(**parameters)
    print(theorist_results)
    # # Run starting from experimentalist
    experimentalist_results = start_experimentalist(**parameters)
    print(experimentalist_results)

    # Runs using run statement
    experiment_runner_results_run = run(
        starting_point="dummy_experiment_runner", **parameters
    )
    print(experiment_runner_results_run)


if __name__ == "__main__":
    test_cycle()
