"""
Test the implementation of using the statemachine package.
https://github.com/fgmacedo/python-statemachine
"""

# !/usr/bin/env python
import pytest  # noqa: 401
from statemachine import State, StateMachine
from statemachine.mixins import MachineMixin

from autora.cycle import (  # noqa: 401
    DataSetCollection,
    run,
    start_experiment_runner,
    start_experimentalist,
    start_theorist,
)


class aer_statemachine(StateMachine):

    # Define states
    new_data = State("new_data", initial=True)
    new_theory = State("new_theory", initial=False)
    new_experiment = State("new_experiment", initial=False)

    # Define transitions
    theorist = new_data.to(new_theory)
    experimentalist = new_theory.to(new_experiment)
    experiment_runner = new_experiment.to(new_data)

    # def on_theorist(self):
    #     print("Running Theorist to get new theory from data.")
    #
    # def on_experimentalist(self):
    #     print("Running Experimentalist to get new test from theory.")
    #
    # def on_experiment_runner(self):
    #     print("Running Experiment Runner to get new data from experiment.")


class aer_cycle(MachineMixin):
    state_machine_name = "aer_statemachine"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()

    def on_theorist(self):
        print("Running Theorist to get new theory from data.")

    def on_experimentalist(self):
        print("Running Experimentalist to get new test from theory.")

    def on_experiment_runner(self):
        print("Running Experiment Runner to get new data from experiment.")


def test_state_class(cycle_limit=5):
    n_cycles = 0
    cycle = aer_cycle(start_value="new_data")

    while n_cycles < cycle_limit:
        if cycle.statemachine.is_new_data:
            cycle.statemachine.run("theorist")
        elif cycle.statemachine.is_new_theory:
            cycle.statemachine.run("experimentalist")
            n_cycles += 1
        elif cycle.statemachine.is_new_experiment:
            cycle.statemachine.run("experiment_runner")


if __name__ == "__main__":
    test_state_class()
    print("end")
