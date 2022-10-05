"""
Test the closed-loop state-machine with run function implementation.

"""

import numpy as np
import pytest  # noqa: 401

from autora.cycle import DataSetCollection, RunCollection, TheoryCollection, run
from autora.variable import Variable, VariableCollection


# Define basic versions of the modules
def dummy_theorist(data, metadata, search_space):
    def theory(x):
        return x + 1

    return theory


def dummy_experimentalist(data, metadata: VariableCollection, theory, n_samples=10):
    low = metadata.independent_variables[0].min
    high = metadata.independent_variables[0].max
    x_prime = np.random.uniform(low, high, n_samples)
    return x_prime


def dummy_experiment_runner(x_prime):
    return x_prime + 1


def test_run_function():
    """
    This is a prototype closed-loop cycle using the "transitions" package as a state-machine
    handler.
    """

    #  Define parameters for run
    # Assuming start from experiment runner we create dummy x' values
    x1 = np.linspace(0, 1, 10)  # Seed x' to input into the experiment runner
    # metadata are value constraints for the experimentalist
    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )
    parameters = RunCollection(
        metadata=metadata,
        search_space=None,
        data=DataSetCollection([]),
        cycle_count=0,
        theories=TheoryCollection([]),
        independent_variable_values=x1,
        max_cycle_count=4,
        first_state="experiment runner",
    )

    results = run(
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        run_container=parameters,
        graph=True,
    )

    assert results.data.datasets.__len__() == results.max_cycle_count, (
        f"Number of datasets generated ({results.data.datasets.__len__()}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )

    assert results.theories.theories.__len__() == results.max_cycle_count, (
        f"Number of theories generated ({results.theories.theories.__len__()}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )

    # Other check ideas
    # Start point checks - need a set-up function


if __name__ == "__main__":
    test_run_function()
    print("end")
