"""
AutoRA full autonomous cycle functions
"""
import logging
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Protocol, Union

from numpy.typing import ArrayLike

from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


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


def combine_datasets(a: Union[DataSetCollection, DataSet], b: DataSet):
    if isinstance(a, DataSet):
        new_collection = DataSetCollection([a, b])
    elif isinstance(a, DataSetCollection):
        new_collection = DataSetCollection(a.datasets + [b])
    return new_collection


SearchSpacePriors = Any


class Theory(Protocol):
    """
    theory(x) -> y

    A Theory maps input values x onto output values y
    """

    def __call__(self, x: IndependentVariableValues) -> DependentVariableValues:
        ...


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

    def __call__(self, x: IndependentVariableValues) -> DependentVariableValues:
        ...


def run(
    theorist: Theorist,
    experimentalist: Experimentalist,
    experiment_runner: ExperimentRunner,
    metadata: VariableCollection,
    search_space: SearchSpacePriors,
    starting_point: Literal["experimentalist", "experiment_runner", "theorist"],
    data: Optional[DataSetCollection] = DataSetCollection([]),
    theory: Optional[Theory] = None,
    independent_variable_values: Optional[IndependentVariableValues] = None,
    cycle_count: int = 0,
):
    if starting_point == "experimentalist":
        start_experimentalist(**locals())
    elif starting_point == "experiment_runner":
        start_experiment_runner(**locals())
    elif starting_point == "theorist":
        start_theorist(**locals())


def new_state(*dicts):
    state = dict()
    for d in dicts:
        state.update(d)
    _logger.info(state)
    return state


def start_theorist(
    theorist: Theorist,
    data: DataSetCollection,
    metadata: VariableCollection,
    search_space: SearchSpacePriors,
    cycle_count: int,
    max_cycle_count: int,
    **kwargs
):
    theory = theorist(data=data, metadata=metadata, search_space=search_space)
    cycle_count += 1
    state = new_state(kwargs, locals())
    if cycle_count == max_cycle_count:
        return state
    return start_experimentalist(**state)


def start_experimentalist(
    experimentalist: Experimentalist,
    data: DataSetCollection,
    metadata: VariableCollection,
    theory: Theory,
    **kwargs
):
    independent_variable_values = experimentalist(
        data=data, metadata=metadata, theory=theory
    )
    state = new_state(kwargs, locals())
    return start_experiment_runner(**state)


def start_experiment_runner(
    experiment_runner: ExperimentRunner,
    independent_variable_values: IndependentVariableValues,
    data: DataSetCollection,
    **kwargs
):
    dependent_variable_values = experiment_runner(x=independent_variable_values)
    data = combine_datasets(
        data,
        DataSet(independent_variable_values, dependent_variable_values),
    )
    state = new_state(kwargs, locals())
    return start_theorist(**state)
