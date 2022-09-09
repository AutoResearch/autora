from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Iterator, Optional, Sequence

import numpy as np


class ValueType(str, Enum):
    """Specifies supported value types supported by Variables."""

    REAL = "real"
    SIGMOID = "sigmoid"
    PROBABILITY = "probability"  # single probability
    PROBABILITY_SAMPLE = "probability_sample"  # sample from single probability
    PROBABILITY_DISTRIBUTION = (
        "probability_distribution"  # probability distribution over classes
    )
    CLASS = "class"  # sample from probability distribution over classes


class Variable:
    """Describes an experimental variable: name, type, range, units, and value of a variable."""

    def __init__(
        self,
        name: str = "",
        value_range: Sequence = (0, 1),
        units: str = "",
        type: ValueType = ValueType.REAL,
        variable_label: str = "",
        rescale: float = 1,
        is_covariate: bool = False,
    ):
        """
        Initialize a variable.

        Args:
            name: name of the variable
            value_range: range of the variable
            units: units of the variable
            type: type of the variable
            variable_label: label of the variable
            rescale: rescale factor for the variable
            is_covariate: whether this variable is a covariate
        """

        self._name = name
        self._units = units
        self._value_range = value_range
        self._value = 0
        self._type = type
        if variable_label == "":
            self._variable_label = self._name
        else:
            self._variable_label = variable_label
        self._rescale = rescale
        self._is_covariate = is_covariate

    @property
    def min(self):
        return self._value_range[0]

    @property
    def max(self):
        return self._value_range[1]

    @property
    def name(self) -> str:
        """Get variable name."""
        return self._name

    @property
    def units(self) -> str:
        """Get variable units.

        Returns:
            variable units
        """
        return self._units

    @property
    def variable_label(self) -> str:
        """Get variable label.

        Returns:
            variable label
        """
        return self._variable_label

    @property
    def is_covariate(self) -> bool:
        """

        Returns:

        """
        return self._is_covariate

    @property
    def type(self) -> ValueType:
        return self._type


@dataclass(frozen=True)
class VariableCollection:
    """Immutable metadata about dependent / independent variables and covariates."""

    independent_variables: Sequence[Variable]
    dependent_variables: Sequence[Variable]
    covariates: Sequence[Variable] = field(default_factory=list)

    @property
    def all_variables(self) -> Iterator[Variable]:
        """Get all variables.

        Returns:
            iterator over all variables
        """
        for vars in (
            self.independent_variables,
            self.dependent_variables,
            self.covariates,
        ):
            for v in vars:
                yield v

    @property
    def variable_names(self) -> Generator[str, None, None]:
        """Get variable names.

        Returns:
            variable names
        """
        return (v.name for v in self.all_variables)

    @property
    def output_type(self) -> str:
        """Get output type.

        Returns:
            output type
        """
        first_type = self.dependent_variables[0].type
        assert all(dv.type == first_type for dv in self.dependent_variables), (
            "Dependent variable output types don't match. "
            "Different output types are not supported yet."
        )
        return first_type

    @property
    def input_dimensions(self) -> int:
        """The number of independent variables and covariates.

        Returns:
            number of independent variables and covariates
        """
        return len(self.independent_variables) + len(self.covariates)

    @property
    def output_dimensions(self) -> int:
        """The number of dependent variables.

        Returns:
            number of dependent variables
        """
        return len(self.dependent_variables)


class IV(Variable):
    """Independent variable."""

    def __init__(self, name="IV", variable_label="Independent Variable", **kwargs):
        """
        Initialize independent variable.

        For arguments, see [autora.variable.Variable][autora.variable.Variable.__init__]
        """
        self._name = name
        self._variable_label = variable_label

        super().__init__(**kwargs)


class DV(Variable):
    """Dependent variable."""

    def __init__(
        self,
        name="DV",
        variable_label="Dependent Variable",
        is_covariate=False,
        **kwargs
    ):
        """
        Initialize dependent variable.

        For arguments, see [autora.variable.Variable][autora.variable.Variable.__init__]
        """
        self._name = name
        self._variable_label = variable_label
        self._is_covariate = is_covariate
        super().__init__(**kwargs)
