"""
Test the implementation of using the transitions package.
https://github.com/pytransitions/transitions
"""

import numpy as np

# !/usr/bin/env python
import pytest  # noqa: 401
from transitions import Machine, State

from autora.cycle import (  # noqa: 401
    DataSetCollection,
    run,
    start_experiment_runner_sm,
    start_experimentalist_sm,
    start_theorist_sm,
)
from autora.variable import Variable, VariableCollection


class aerCycle(object):
    """
    Contains attributes/parameters of the AER cycle and sets up methods to call upon entering and
    exiting states of the cycle.

    TODO
    1. Currently takes all keywords and kwargs dicts passed in. Should probably have specified
    assignments to check if all the correct parameters are supplied and set defaults values.
    """

    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def on_enter_theorist(self):
        print("Running Theorist to get new theory from data.")
        start_theorist_sm(self)

    def on_exit_theorist(self):
        print("Packaging theory for experimentalist.")

    def on_enter_experimentalist(self):
        print("Running Experimentalist to get new test from theory.")
        start_experimentalist_sm(self)

    def on_exit_experimentalist(self):
        print("Validating Experimental parameters.")

    def on_enter_experiment_runner(self):
        print("Running Experiment Runner to get new data from experiment parameters.")
        start_experiment_runner_sm(self)

    def on_exit_experiment_runner(self):
        print("Packaging collected data for theorist.")

    def on_enter_sleep(self):
        print("Sleeping...")


class aerMachine(object):
    """
    Initializes state machine using defined states and transitions and supplied aerCycle model.
    """

    def __init__(self, model):
        # Define states
        states = [
            State(
                name="experiment_runner",
                on_enter=["on_enter_experiment_runner"],
                on_exit=["on_exit_experiment_runner"],
            ),
            State(
                name="theorist",
                on_enter=["on_enter_theorist"],
                on_exit=["on_exit_theorist"],
            ),
            State(
                name="experimentalist",
                on_enter=["on_enter_experimentalist"],
                on_exit=["on_exit_experimentalist"],
            ),
            State(name="sleep", on_enter=["on_enter_experimentalist"]),
        ]

        transitions = [
            {
                "trigger": "run_theorist",
                "source": ["experiment_runner", "sleep"],
                "dest": "theorist",
            },
            {
                "trigger": "run_experimentalist",
                "source": ["theorist", "sleep"],
                "dest": "experimentalist",
            },
            {
                "trigger": "run_experiment_runner",
                "source": ["experimentalist", "sleep"],
                "dest": "experiment_runner",
            },
            {
                "trigger": "sleep",
                "source": ["experiment_runner", "theorist", "experimentalist"],
                "dest": "sleep",
            },
        ]

        self.machine = Machine(
            model=model,
            states=states,
            transitions=transitions,
            initial="sleep",
            queued=True,
            ignore_invalid_triggers=False,
        )


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


def test_state_class():
    """
    This is a prototype closed-loop cycle using the "transitions" package as a state-machine
    handler.
    """

    # First state to move to from initial sleep state
    first_state = "experiment runner"

    #  Define parameters for run
    # Assuming start from experiment runner we create dummy x' values
    x1 = np.linspace(0, 1, 10)  # Seed x' to input into the experiment runner
    # metadata are value constraints for the experimentalist
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
        max_cycle_count=4,
    )

    # Initialize the cycle model
    cycle = aerCycle(**parameters)
    # Initialize state machine using model
    aerMachine(model=cycle)

    while cycle.cycle_count < cycle.max_cycle_count:

        # First iteration transitions from sleep to the first state
        if (cycle.cycle_count == 0) & cycle.is_sleep():
            if first_state.lower() == "theorist":
                cycle.run_theorist()
            elif first_state.lower() == "experimentalist":
                cycle.run_experimentalist()
            elif first_state.lower() == "experiment runner":
                cycle.run_experiment_runner()
        # Subsequent iterations will cycle by state
        elif cycle.is_experiment_runner():
            cycle.run_theorist()

        elif cycle.is_theorist():
            cycle.run_experimentalist()

        elif cycle.is_experimentalist():
            cycle.run_experiment_runner()

    print(f"{cycle.data.datasets.__len__()} datasets generated.")

    assert cycle.data.datasets.__len__() == cycle.max_cycle_count, (
        f"Number of datasets generated ({cycle.data.datasets.__len__()}) "
        f"should equal the max number of cycles ({cycle.max_cycle_count})."
    )


if __name__ == "__main__":
    test_state_class()
    print("end")
