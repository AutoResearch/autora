"""
Test the closed-loop state-machine with run function implementation.

"""

import numpy as np

from autora.cycle import AERCycle
from autora.variable import Variable, VariableCollection


# Define basic versions of the modules
def dummy_theorist(data, metadata, search_space):
    def theory(x):
        return x + 1

    return theory


def dummy_experimentalist(data, metadata: VariableCollection, theory, n_samples=10):

    assert metadata.independent_variables[0].value_range is not None

    low = min(metadata.independent_variables[0].value_range)
    high = max(metadata.independent_variables[0].value_range)
    x_prime = np.random.uniform(low, high, n_samples)
    return x_prime


def dummy_experiment_runner(x_prime):
    return x_prime.x + 1


def test_run():
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

    cycle = AERCycle(
        theorist=dummy_theorist,
        experimentalist=dummy_experimentalist,
        experiment_runner=dummy_experiment_runner,
        metadata=metadata,
        first_state="experiment runner",
        independent_variable_values=x1,
        add_graphing=True,
    )
    cycle.run()
    results = cycle.results

    # Plot diagram, will remove this functionality in production
    cycle.get_graph().draw("state_machine_diagram.png", prog="dot")

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
