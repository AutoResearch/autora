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

    x1 = np.linspace(0, 1, 100)

    def experiment_runner(x):
        return x + 1

    def dummy_theorist(data, metadata, search_space):
        def theory(x):
            return x + 1

        return theory

    def dummy_experimentalist(data, metadata: VariableCollection, theory):
        low = metadata.independent_variables[0].min
        high = metadata.independent_variables[0].max
        x_prime = np.random.uniform(low, high, 10)
        return x_prime

    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )

    parameters = dict(
        cycle_count=0,
        max_cycle_count=10,
        experiment_runner=experiment_runner,
        independent_variable_values=x1,
        data=DataSetCollection([]),
        theorist=dummy_theorist,
        search_space=None,
        metadata=metadata,
        experimentalist=dummy_experimentalist,
        theory=None,
    )
    experiment_runner_results = start_experiment_runner(**parameters)
    print(experiment_runner_results)
    theorist_results = start_theorist(**parameters)
    print(theorist_results)
    experimentalist_results = start_experimentalist(**parameters)
    print(experimentalist_results)

    experiment_runner_results_run = run(
        starting_point="experiment_runner", **parameters
    )
    print(experiment_runner_results_run)


if __name__ == "__main__":
    test_cycle()
