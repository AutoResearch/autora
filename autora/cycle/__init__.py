"""
AutoRA full autonomous cycle functions
"""
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
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


class ExperimentalDesign(Protocol):
    """
    Transformation of independent variable values
    given restrictions specified by the experimental design.

    Implementation ideas:
        - Could be a callable passed to an experimentalist OR
        - Defined as part of a pipeline after the Experimentalist
    """

    def __call__(
        self, ivs: IndependentVariableValues, metadata: VariableCollection
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


class AERModule(Enum):
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
        run_container: RunCollection,
    ):
        self.theorist = theorist
        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner
        self.run_container = run_container

    def is_ready_for_experiment_runner(self):
        return self.run_container.first_state is AERModule.EXPERIMENT_RUNNER

    def is_ready_for_theorist(self):
        return self.run_container.first_state is AERModule.THEORIST

    def is_ready_for_experimentalist(self):
        return self.run_container.first_state is AERModule.EXPERIMENTALIST

    def is_max_cycles(self):
        return self.run_container.cycle_count >= self.run_container.max_cycle_count

    def start_theorist(self):
        print("Running Theorist to get new theory from data.")
        # Run Theorist
        new_theory = self.theorist(
            data=self.run_container.data,
            metadata=self.run_container.metadata,
            search_space=self.run_container.search_space,
        )
        # Append new theory to theory list
        new_theory_collection = combine_theories(
            self.run_container.theories, new_theory
        )

        # Update run container with updated theory and cycle count
        self.run_container = update_run_collection(
            self.run_container,
            theories=new_theory_collection,
            cycle_count=self.run_container.cycle_count + 1,
        )

    def start_experimentalist(self):
        print("Running Experimentalist to get new test from theory.")
        # Run experimentalist
        new_independent_variable_values = self.experimentalist(
            data=self.run_container.data,
            metadata=self.run_container.metadata,
            theory=self.run_container.theories.theories[-1],
        )
        # Update run container with X' values to test
        self.run_container = update_run_collection(
            self.run_container,
            independent_variable_values=new_independent_variable_values,
        )

    def start_experiment_runner(self):
        print("Running Experiment Runner to get new data from experiment parameters.")
        # Run experiment runner
        dependent_variable_values = self.experiment_runner(
            x_prime=self.run_container.independent_variable_values
        )
        # Append or create a new dataset object
        new_dataset_collection = combine_datasets(
            self.run_container.data,
            DataSet(
                self.run_container.independent_variable_values,
                dependent_variable_values,
            ),
        )

        # Update run container with new DataSetCollection
        self.run_container = update_run_collection(
            self.run_container,
            data=new_dataset_collection,
        )

    def end_statement(self):
        print("At end state.")


def create_state_machine(model, graph=False):
    """
    Adds state-machine methods to a cycle model.
    Args:
        model (AERCycle): Class with run attributes, module handler methods,
                          and node transition conditions defined.
        graph (bool): Option add graphing methods to the  model.

    Returns:
    AERCycle model with state-machine methods.
    """
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


def run(
    theorist: Theorist,
    experimentalist: Experimentalist,
    experiment_runner: ExperimentRunner,
    **kwargs
):
    """
    Runs the closed-loop prototype.
    Args:
        theorist: Theorist function
        experimentalist: Experimentalist function
        experiment_runner: Experiment runner function
        run_container: RunCollection containing parameters and containers to store data
        graph (bool): Option to print state-machine diagram

    Returns:
    The run container object with run parameters and aggregated data.
    """
    # Initialize the data
    run_container = RunCollection(**kwargs)
    # Initialize the cycle model
    cycle = AERCycle(theorist, experimentalist, experiment_runner, run_container)
    # Initialize state machine using model. This applies state machine methods to the model.
    cycle = create_state_machine(cycle, graph=True)

    # Run the state machine
    while True:
        try:
            cycle.next_step()  # type: ignore
        except core.MachineError:
            break

    # Return run container
    return cycle.run_container
