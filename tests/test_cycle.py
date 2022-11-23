"""
Test the closed-loop state-machine with run function implementation.

"""

import numpy as np
import pytest

from autora.cycle import AERCycle, DataSet
from autora.variable import Variable, VariableCollection

# Define a simple truth we wish to recover
def ground_truth(x):
    return x + 1


# Define basic versions of the modules
def dummy_theorist(data, metadata, search_space):
    def theory(x):
        return x + 1

    return theory


def zeroth_theory(x):
    return 0 * x


def dummy_experimentalist(data, metadata: VariableCollection, theory, n_samples=10):

    assert metadata.independent_variables[0].value_range is not None

    low = min(metadata.independent_variables[0].value_range)
    high = max(metadata.independent_variables[0].value_range)
    x_prime = np.random.uniform(low, high, n_samples)
    return x_prime


def dummy_experiment_runner(x_prime):
    return ground_truth(x_prime.x)


@pytest.fixture
def metadata():
    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", value_range=(-5, 5))],
        dependent_variables=[Variable(name="y", value_range=(-10, 10))],
    )
    return metadata


def test_run_starting_at_experimentalist(metadata):
    """
    This is a prototype closed-loop cycle using the "transitions" package as a state-machine
    handler.
    """

    #  Define parameters for run
    cycle = AERCycle(
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        first_state="experimentalist",
        theories=[zeroth_theory],
    )
    cycle.run()
    results = cycle.results

    assert len(results.data.datasets) == results.max_cycle_count, (
        f"Number of datasets generated ({len(results.data.datasets)}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )

    assert len(results.theories.theories) == results.max_cycle_count + 1, (
        f"Number of theories generated ({len(results.theories.theories)}) "
        f"should equal the max number of cycles ({results.max_cycle_count}) "
        f"plus one because we had a 'seed' initial theory."
    )


def test_run_starting_at_experiment_runner(metadata):
    """
    This is a prototype closed-loop cycle using the "transitions" package as a state-machine
    handler.
    """

    #  Define parameters for run
    # Assuming start from experiment runner we create dummy x' values
    x1 = np.linspace(0, 1, 10)  # Seed x' to input into the experiment runner

    cycle = AERCycle(
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        first_state="experiment runner",
        independent_variable_values=x1,
    )
    cycle.run()
    results = cycle.results

    assert len(results.data.datasets) == results.max_cycle_count, (
        f"Number of datasets generated ({len(results.data.datasets)}) "
        f"should equal the max number of cycles ({len(results.max_cycle_count)})."
    )

    assert len(results.theories.theories) == results.max_cycle_count, (
        f"Number of theories generated ({len(results.theories.theories)}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )


def test_run_starting_at_theorist(metadata):
    """
    This is a prototype closed-loop cycle using the "transitions" package as a state-machine
    handler.
    """

    #  Define parameters for run
    # Assuming start from experiment runner we create dummy x' values
    x0 = np.linspace(0, 1, 10)  # Seed x' to input into the experiment runner
    y0 = ground_truth(x0)
    data = [DataSet(x0, y0)]

    cycle = AERCycle(
        max_cycle_count=5,
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        first_state="theorist",
        data=data,
    )
    cycle.run()
    results = cycle.results

    assert len(results.data.datasets) == results.max_cycle_count, (
        f"Number of datasets generated ({len(results.data.datasets)}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )

    assert len(results.theories.theories) == results.max_cycle_count, (
        f"Number of theories generated ({len(results.theories.theories)}) "
        f"should equal the max number of cycles ({results.max_cycle_count})."
    )


def test_graphing(metadata):
    cycle = AERCycle(
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        add_graphing=True,
        first_state="experimentalist",
    )

    # Plot diagram, will remove this functionality in production
    cycle.get_graph().draw("state_machine_diagram.png", prog="dot")
