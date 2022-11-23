"""
AutoRA full autonomous cycle functions
"""
import logging
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Number
from typing import List, Protocol, Union

from numpy.typing import ArrayLike
from transitions import Machine, State, core
from transitions.extensions import GraphMachine

from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSpacePriors:
    priors: List[str]


@dataclass(frozen=True)
class IndependentVariableValues:
    x: ArrayLike


@dataclass(frozen=True)
class DependentVariableValues:
    y: ArrayLike


@dataclass(frozen=True)
class DataSet:
    x: IndependentVariableValues
    y: DependentVariableValues


@dataclass(frozen=False)
class DataSetCollection:
    datasets: List[DataSet]

    def __getitem__(self, item):
        return self.datasets[item]


def combine_datasets(a: Union[DataSetCollection, DataSet], b: DataSet):
    if isinstance(a, DataSet):
        new_collection = DataSetCollection([a, b])
    elif isinstance(a, DataSetCollection):
        new_collection = DataSetCollection(a.datasets + [b])
    return new_collection


class Theory(Protocol):
    """
    theory(x) -> y

    A Theory maps input values x onto output values y
    """

    def __call__(self, x: IndependentVariableValues) -> DependentVariableValues:
        ...


@dataclass(frozen=False)
class TheoryCollection:
    theories: List[Theory]

    def __getitem__(self, item):
        return self.theories[item]


def combine_theories(a: Union[TheoryCollection, Theory], b: Theory):
    if callable(a):
        new_collection = TheoryCollection([a, b])
    elif isinstance(a, TheoryCollection):
        new_collection = TheoryCollection(a.theories + [b])
    return new_collection


class Theorist(Protocol):
    """
    theorist(x, y, metadata) -> [theory(x′) -> y′]

    A Theorist takes values of x and y and finds a theory f(x) ~> y
    """

    def __call__(
        self,
        data: DataSetCollection,
        metadata: VariableCollection,
        search_space: SearchSpacePriors,
    ) -> Theory:
        ...


class Experimentalist(Protocol):
    """
    experimentalist(x, y, metadata, theory) -> x′

    An experimentalist takes known values of x and y, along with metadata about their domains, and
    the best known theory, and suggests new x′ which can be used in new experiments to generate
    new data.
    """

    def __call__(
        self, data: DataSetCollection, metadata: VariableCollection, theory: Theory
    ) -> IndependentVariableValues:
        ...


class ExperimentRunner(Protocol):
    """
    experiment_runner(x) -> y

    An experiment_runner takes some independent variable values x
    and generates new observations y for those.
    """

    def __call__(self, x_prime: IndependentVariableValues) -> DependentVariableValues:
        ...


class AERModule(str, Enum):
    THEORIST = "theorist"
    EXPERIMENTALIST = "experimentalist"
    EXPERIMENT_RUNNER = "experiment runner"


@dataclass(frozen=False)
class RunCollection:
    # Static
    metadata: VariableCollection
    first_state: AERModule
    max_cycle_count: int
    search_space: SearchSpacePriors

    # Updates each cycle
    independent_variable_values: IndependentVariableValues
    cycle_count: int

    # Aggregates each cycle
    data: DataSetCollection
    theories: TheoryCollection


def update_run_collection(run_collection: RunCollection, **changes):
    new_run_collection = replace(run_collection, **changes)
    return new_run_collection


class AERCycle(object):
    """
    Contains attributes/parameters of the AER cycle and handler and condition methods
    to enact between state transitions.



    """

    def __init__(
        self,
        theorist: Theorist,
        experimentalist: Experimentalist,
        experiment_runner: ExperimentRunner,
        metadata: VariableCollection,
        first_state: str,
        search_space: List[str] = [],
        max_cycle_count: int = 5,
        cycle_count: int = 0,
        independent_variable_values: List[Number] = [],
        data: DataSetCollection = DataSetCollection([]),
        theories: TheoryCollection = TheoryCollection([]),
        name=None,
        add_graphing=True,
    ):
        """

        Args:
        ---Required---
        theorist: Theorist function
        experimentalist: Experimentalist function
        experiment_runner: Experiment runner function
        metadata: VariableCollection containing attributes of each independent and dependent
            variable
        first_state: Starting module of AER Cycle. Options: "theorist", "experimentalist",
            "experiment runner"

        ---Optional---
        search_space: Priors
        max_cycle_count: Maximum number of cycles to conduct. Aggregates after the Theorist.
        cycle_count: Current cycle number, default = 0
        add_graphing: Boolean to add get_graph method to graph state machine
        name: Name of the cycle

        ---Optional Depending on First State---
        independent_variable_values: Seed values for the "experiment runner" if starting at the
            "experiment runner" module.
        data: Seed dataset values if starting at the theorist or experimentalist. Default is an
            empty DataSetCollection.
        theories: Seed theory if starting at the experimentalist. Default is an empty
            TheoryCollection.
        """

        # Create a run collection object
        run_container = RunCollection(
            metadata=metadata,
            first_state=AERModule(first_state),
            search_space=SearchSpacePriors(search_space),
            max_cycle_count=max_cycle_count,
            cycle_count=cycle_count,
            independent_variable_values=IndependentVariableValues(
                independent_variable_values
            ),
            data=data,
            theories=theories,
        )

        self._theorist = theorist
        self._experimentalist = experimentalist
        self._experiment_runner = experiment_runner
        self._run_container = run_container
        self.name = name if name is not None else "AERCycle"

        # Add machine methods
        self._states = self.__add_states()
        self._transitions = self.__add_transitions()
        self.__add_sm_methods(graph=add_graphing)

    def is_ready_for_experiment_runner(self):
        return AERModule(self._run_container.first_state) is AERModule.EXPERIMENT_RUNNER

    def is_ready_for_theorist(self):
        return AERModule(self._run_container.first_state) is AERModule.THEORIST

    def is_ready_for_experimentalist(self):
        return AERModule(self._run_container.first_state) is AERModule.EXPERIMENTALIST

    def is_max_cycles(self):
        return self._run_container.cycle_count >= self._run_container.max_cycle_count

    def start_theorist(self):
        print("Running Theorist to get new theory from data.")
        # Run Theorist
        new_theory = self._theorist(
            data=self._run_container.data,
            metadata=self._run_container.metadata,
            search_space=self._run_container.search_space,
        )
        # Append new theory to theory list
        new_theory_collection = combine_theories(
            self._run_container.theories, new_theory
        )

        # Update run container with updated theory and cycle count
        self._run_container = update_run_collection(
            self._run_container,
            theories=new_theory_collection,
            cycle_count=self._run_container.cycle_count + 1,
        )

    def start_experimentalist(self):
        print("Running Experimentalist to get new test from theory.")
        # Run experimentalist
        new_independent_variable_values = self._experimentalist(
            data=self._run_container.data,
            metadata=self._run_container.metadata,
            theory=self._run_container.theories[-1],
        )
        # Update run container with X' values to test
        self._run_container = update_run_collection(
            self._run_container,
            independent_variable_values=IndependentVariableValues(
                new_independent_variable_values
            ),
        )

    def start_experiment_runner(self):
        print("Running Experiment Runner to get new data from experiment parameters.")
        # Run experiment runner
        dependent_variable_values = self._experiment_runner(
            x_prime=self._run_container.independent_variable_values
        )
        # Append or create a new dataset object
        new_dataset_collection = combine_datasets(
            self._run_container.data,
            DataSet(
                self._run_container.independent_variable_values,
                dependent_variable_values,
            ),
        )

        # Update run container with new DataSetCollection
        self._run_container = update_run_collection(
            self._run_container,
            data=new_dataset_collection,
        )

    def __end_statement(self):
        print("At end state.")

    def __add_states(self):
        l_states = [
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
            State(name="end", on_enter=["_AERCycle__end_statement"]),
        ]
        return l_states

    def __add_transitions(self):
        l_transitions = [
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

        return l_transitions

    def __add_sm_methods(self, graph):
        # Initialize state machine using model. This applies state machine methods to the model.

        if not graph:
            Machine(
                model=self,
                states=self._states,
                transitions=self._transitions,
                initial="start",
                queued=True,
                ignore_invalid_triggers=False,
            )
        else:

            GraphMachine(
                model=self,
                states=self._states,
                transitions=self._transitions,
                initial="start",
                queued=True,
                ignore_invalid_triggers=False,
                show_state_attributes=True,
                show_conditions=True,
            )

    def run(self):
        # Run the state machine
        while True:
            try:
                self.next_step()  # type: ignore
            except core.MachineError:
                break
        print("Run finished.")

    @property
    def results(self):
        return self._run_container
