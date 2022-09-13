"""
AutoRA full autonomous cycle functions
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Type

import numpy as np
from numpy.typing import ArrayLike

from autora.variable import VariableCollection


@dataclass(frozen=True)
class IndependentVariableValues:
    x: ArrayLike


@dataclass(frozen=True)
class DependentVariableValues:
    y: ArrayLike


@dataclass(frozen=True)
class DataSet:
    x: IndependentVariableValues
    y: IndependentVariableValues


@dataclass(frozen=False)
class DataSetCollection:
    datasets: List[DataSet]


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


# def run(
#     theorist: Theorist,
#     experimentalist: Experimentalist,
#     experiment_runner: ExperimentRunner,
#     metadata: VariableCollection,
#     seed_data: Optional[DataSetCollection] = None,
#     search_space: ? = None
# ):
#     """
#
#     Args:
#         theorist:
#         experimentalist:
#         experiment_runner:
#
#     Returns:
#
#     """
#     if seed_data is None:
#         seed_independent_variable_values = experimentalist(
#             data=None,
#             metadata=metadata
#         )
#         seed_data_ = experiment_runner(seed_independent_variable_values)
#     else:
#         seed_data_ = seed_data
#
#     theory = theorist(seed_data_, metadata, search_space)
#
#     new_independent_variable_values = experimentalist(
#         data=seed_data_,
#         theory=theory,
#         metadata=metadata,
#     )
#
#     new_dependent_variable_values = experiment_runner(
#         new_independent_variable_values
#     )
#
#     ... go back to the theorist and start over
#
#
#
#
#
#
