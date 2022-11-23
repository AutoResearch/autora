"""
Test the implementation of using the transitions package.
https://github.com/pytransitions/transitions
"""

import numpy as np

# !/usr/bin/env python
import pytest  # noqa: 401
from transitions import Machine, State, core
from transitions.extensions import GraphMachine

from autora.cycle import DataSet, DataSetCollection, combine_datasets  # noqa: 401
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
    return x_prime + 1


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

    def is_ready_for_experiment_runner(self):
        if self.first_state.lower() == "experiment runner":
            return True
        else:
            return False

    def is_ready_for_theorist(self):
        if self.first_state.lower() == "theorist":
            return True
        else:
            return False

    def is_ready_for_experimentalist(self):
        if self.first_state.lower() == "experimentalist":
            return True
        else:
            return False

    def is_max_cycles(self):
        if self.cycle_count == self.max_cycle_count:
            return True
        else:
            return False

    def start_theorist(self):
        print("Running Theorist to get new theory from data.")
        self.theory = self.theorist(
            data=self.data,
            metadata=self.metadata,
            search_space=self.search_space,
        )
        self.cycle_count += 1
        print(f"{self.cycle_count}/{self.max_cycle_count} cycles")
        return self

    def start_experimentalist(self):
        print("Running Experimentalist to get new test from theory.")
        self.independent_variable_values = self.experimentalist(
            data=self.data, metadata=self.metadata, theory=self.theory
        )
        return self

    def start_experiment_runner(self):
        print("Running Experiment Runner to get new data from experiment parameters.")
        dependent_variable_values = self.experiment_runner(
            x_prime=self.independent_variable_values
        )
        self.data = combine_datasets(
            self.data,
            DataSet(
                self.independent_variable_values,
                dependent_variable_values,
            ),
        )
        return self

    def end_statement(self):
        print("At end state.")


def create_state_machine(model, graph=False):

    states = [
        State(name="start"),
        State(
            name="experiment_runner",
            on_enter=["start_experiment_runner"],
        ),
        State(
            name="theorist",
            on_enter=["start_theorist"],
        ),
        State(
            name="experimentalist",
            on_enter=["start_experimentalist"],
        ),
        State(name="end", on_enter=["end_statement"]),
    ]

    transitions = [
        # Start transitions
        {
            "trigger": "next_step",
            "source": ["start"],
            "dest": "experiment_runner",
            "conditions": ["is_ready_for_experiment_runner"],
        },
        {
            "trigger": "next_step",
            "source": ["start"],
            "dest": "theorist",
            "conditions": ["is_ready_for_theorist"],
        },
        {
            "trigger": "next_step",
            "source": ["start"],
            "dest": "experimentalist",
            "conditions": ["is_ready_for_experimentalist"],
        },
        # Module to module transitions
        {
            "trigger": "next_step",
            "source": ["experiment_runner"],
            "dest": "theorist",
            "unless": [],
        },
        {
            "trigger": "next_step",
            "source": ["theorist"],
            "dest": "experimentalist",
            "unless": ["is_max_cycles"],
        },
        {
            "trigger": "next_step",
            "source": ["experimentalist"],
            "dest": "experiment_runner",
            "unless": [],
        },
        # End transition
        {
            "trigger": "next_step",
            "source": ["theorist"],
            "dest": "end",
            "conditions": ["is_max_cycles"],
        },
    ]

    if graph:
        GraphMachine(
            model=model,
            states=states,
            transitions=transitions,
            initial="start",
            queued=True,
            ignore_invalid_triggers=False,
            show_state_attributes=True,
            show_conditions=True,
        )
    else:
        Machine(
            model=model,
            states=states,
            transitions=transitions,
            initial="start",
            queued=True,
            ignore_invalid_triggers=False,
        )

    return model


def test_state_class():
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
        first_state="experiment runner",
    )

    # Initialize the cycle model
    cycle = aerCycle(**parameters)
    # Initialize state machine using model. This applies state machine methods to the model.
    cycle = create_state_machine(cycle, graph=True)
    # Run the state machine
    while True:
        try:
            cycle.next_step()
        except core.MachineError:
            break

    # Save diagram of the state machine
    cycle.get_graph().draw("state_diagram.png", prog="dot")

    # return a dataclass

    assert cycle.data.datasets.__len__() == cycle.max_cycle_count, (
        f"Number of datasets generated ({cycle.data.datasets.__len__()}) "
        f"should equal the max number of cycles ({cycle.max_cycle_count})."
    )

    # Other check ideas
    # Start point checks - need a set-up function


if __name__ == "__main__":
    test_state_class()
    print("end")
