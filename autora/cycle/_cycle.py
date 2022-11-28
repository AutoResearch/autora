from dataclasses import dataclass, replace
from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from autora.cycle._state import State, StateMachine, Transition
from autora.variable import VariableCollection


@dataclass(frozen=True)
class _CycleOwnStateMachineRunCollection:
    # Static
    metadata: VariableCollection

    # Updates each cycle
    max_cycle_count: int
    cycle_count: int

    # Aggregates each cycle from the:
    # ... Experimentalist
    independent_variable_values: List[np.ndarray]
    # ... Experiment Runner
    data: List[np.ndarray]
    # ... Theorist
    theories: List[BaseEstimator]


def update_run_collection(
    run_collection: _CycleOwnStateMachineRunCollection, **changes
):
    new_run_collection = replace(run_collection, **changes)
    return new_run_collection


class _CycleOwnStateMachine:
    def __init__(
        self, metadata, theorist, experimentalist, experiment_runner, max_cycle_count
    ):
        """

        Args:
            theorist:
            experimentalist:
            experiment_runner:

        Examples:
            Aim: Use the Cycle to recover a simple ground truth theory from noisy data.

            >>> def ground_truth(x):
            ...     return x + 1

            The space of allowed x values is the integers between 0 and 10 inclusive,
            and we record the allowed output values as well.
            >>> from autora.variable import VariableCollection, Variable
            >>> metadata = VariableCollection(
            ...    independent_variables=[Variable(name="x1", allowed_values=range(11))],
            ...    dependent_variables=[Variable(name="y", value_range=(-20, 20))],
            ...    )

            When we run a synthetic experiment, we get a reproducible noisy result:
            >>> import numpy as np
            >>> def get_example_synthetic_experiment_runner():
            ...     rng = np.random.default_rng(seed=180)
            ...     def runner(x):
            ...         return ground_truth(x) + rng.normal(0, 0.1, x.shape)
            ...     return runner
            >>> example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()
            >>> example_synthetic_experiment_runner(np.ndarray([1]))
            array([2.04339546])

            The experimentalist is used to propose experiments.
            Since the space of values is so restricted, we can just sample them all each time.
            >>> from autora.experimentalist.pipeline import make_pipeline
            >>> from autora.experimentalist.pool import grid_pool
            >>> example_experimentalist = make_pipeline([grid_pool(metadata.independent_variables)])

            The theorist "tries" to work out the best theory.
            We use a trivial scikit-learn regressor.
            >>> from sklearn.linear_model import LinearRegression
            >>> example_theorist = LinearRegression()

            We initialize the Cycle with the theorist, experimentalist and experiment runner,
            and define the maximum cycle count.
            >>> cycle = _CycleOwnStateMachine(
            ...     metadata=metadata,
            ...     theorist=example_theorist,
            ...     experimentalist=example_experimentalist,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     max_cycle_count=10,
            ... )
            >>> cycle # doctest: +ELLIPSIS
            <_cycle._CycleOwnStateMachine object at 0x...>

            We can run the cycle by calling the run method:
            >>> cycle.run()
        """
        self._machine = self._create_state_machine()

        self.theorist = theorist
        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner

        self.data = _CycleOwnStateMachineRunCollection(
            metadata=metadata,
            max_cycle_count=max_cycle_count,
            independent_variable_values=[],
            data=[],
            theories=[],
        )

    def run(self):
        while True:
            try:
                self._machine.step()
            except StopIteration:
                return

    def _create_state_machine(self):
        machine = StateMachine()

        state_start = State("start")
        state_theorist = State("theorist", callback=self._theorist_callback)
        state_experimentalist = State(
            "experimentalist", callback=self._experimentalist_callback
        )
        state_experiment_runner = State(
            "experiment runner", callback=self._experiment_runner_callback
        )
        state_end = State("end")
        machine.states.append(state_start)
        machine.states.append(state_theorist)
        machine.states.append(state_experimentalist)
        machine.states.append(state_experiment_runner)
        machine.states.append(state_end)
        transition_start_experimentalist = Transition(
            state1=state_start,
            state2=state_experimentalist,
        )
        transition_start_experiment_runner = Transition(
            state1=state_start, state2=state_experiment_runner
        )
        transition_start_theorist = Transition(
            state1=state_start, state2=state_theorist
        )
        machine.transitions.append(transition_start_experimentalist)
        machine.transitions.append(transition_start_experiment_runner)
        machine.transitions.append(transition_start_theorist)
        transition_theorist_experimentalist = Transition(
            state1=state_theorist, state2=state_experimentalist
        )
        transition_experimentalist_experiment_runner = Transition(
            state1=state_experimentalist,
            state2=state_experiment_runner,
        )
        transition_experiment_runner_theorist = Transition(
            state1=state_experiment_runner,
            state2=state_theorist,
        )
        machine.transitions.append(transition_theorist_experimentalist)
        machine.transitions.append(transition_experimentalist_experiment_runner)
        machine.transitions.append(transition_experiment_runner_theorist)
        transition_experimentalist_end = Transition(
            state1=state_experimentalist,
            state2=state_end,
        )
        transition_experiment_runner_end = Transition(
            state1=state_experiment_runner,
            state2=state_end,
        )
        transition_theorist_end = Transition(
            state1=state_theorist,
            state2=state_end,
        )
        machine.transitions.append(transition_experimentalist_end)
        machine.transitions.append(transition_experiment_runner_end)
        machine.transitions.append(transition_theorist_end)

        machine.current_state = state_start

        return machine

    def _theorist_callback(self):
        self.data = replace(self.data, cycle_count=(self.data.cycle_count + 1))
        return

    def _experimentalist_callback(self):
        return

    def _experiment_runner_callback(self):
        return

    @staticmethod
    def _stopping_condition(data: _CycleOwnStateMachineRunCollection):
        return data.cycle_count > data.max_cycle_count
