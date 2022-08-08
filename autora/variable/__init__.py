from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

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
        name="",
        value_range=(0, 1),
        units="",
        type=ValueType.REAL,
        variable_label="",
        rescale=1,
        is_covariate=False,
        participant=None,
    ):

        self._name = name
        self._units = units
        self._value_range = value_range
        self._value = 0
        self.type = type
        if variable_label == "":
            self._variable_label = self._name
        else:
            self._variable_label = variable_label
        self._rescale = rescale
        self._is_covariate = is_covariate
        self._participant = participant

    def __get_value_range__(self):
        """Get range of variable.
        The variable range determines the minimum and maximum allowed value
        to be manipulated or measured."""
        return self._value_range

    def __set_value_range__(self, value_range):
        """Set range of variable.
        The variable range determines the minimum and maximum allowed value
        to be manipulated or measured."""
        self._value_range = value_range

    def __cap_value__(self, value):
        """Cap value of variable."""
        minimum = self._value_range[0]
        maximum = self._value_range[1]
        return np.min([np.max([value, minimum]), maximum])

    def get_value(self):
        """Get value."""
        return self._value * self._rescale

    def set_value(self, value):
        """Set value."""
        self._value = self.__cap_value__(value)

    def get_value_from_dict(self, dictionary, position):
        """Reads value of independent variable from a dictionary."""

        value_list = dictionary[self.get_name()]

        if position > len(value_list):
            raise Exception(
                f"Queried position "
                f"{str(position)}"
                f" for variable "
                f"{self.get_name()}"
                f"'exceeds number of available positions for that variable in the dictionary."
            )

        return value_list[position] * self._rescale

    def get_value_list_from_dict(self, dictionary):
        """Gets the rescaled values of independent variables from a dictionary."""
        value_list = dictionary[self.get_name()]

        rescaled_list = [element * self._rescale for element in value_list]

        return rescaled_list

    def set_value_from_dict(self, dictionary, position):
        """
        Reads and sets value of independent variable from a dictionary
        with variable_label being the key
        """

        value_list = dictionary[self.get_name()]

        if position > len(value_list):
            raise Exception(
                f"Queried position "
                f"{str(position)}"
                f" for variable "
                f"{self.get_name()}"
                f" exceeds number of available positions for that variable in the dictionary."
            )

        self.set_value(value_list[position])

    def get_name(self):
        """Get variable name."""
        return self._name

    def set_name(self, name):
        """Set variable name."""
        self._name = name

    def get_units(self):
        """Get variable units."""
        return self._units

    def set_units(self, units):
        """Set variable units."""
        self._units = units

    def get_variable_label(self):
        """Get variable label."""
        return self._variable_label

    def set_variable_label(self, variable_label):
        """Set variable label."""
        self._variable_label = variable_label

    def set_covariate(self, is_covariate):
        """Set whether this dependent variable is treated as covariate."""
        self._is_covariate = is_covariate


@dataclass(frozen=True)
class VariableCollection:
    """Immutable metadata about dependent / independent variables and covariates."""

    independent_variables: Sequence[Variable]
    dependent_variables: Sequence[Variable]
    covariates: Sequence[Variable] = field(default_factory=list)

    @property
    def all_variables(self) -> Iterator[Variable]:
        for vars in (
            self.independent_variables,
            self.dependent_variables,
            self.covariates,
        ):
            for v in vars:
                yield v

    @property
    def variable_names(self):
        return (v.get_name() for v in self.all_variables)

    @property
    def output_type(self):
        first_type = self.dependent_variables[0].type
        assert all(dv.type == first_type for dv in self.dependent_variables), (
            "Dependent variable output types don't match. "
            "Different output types are not supported yet."
        )
        return first_type

    @property
    def input_dimensions(self):
        """The number of independent variables and covariates."""
        return len(self.independent_variables) + len(self.covariates)

    @property
    def output_dimensions(self):
        """The number of dependent variables."""
        return len(self.dependent_variables)


class IV(Variable):
    def __init__(self, *args, **kwargs):
        self._name = "IV"
        self._variable_label = "Independent Variable"

        super().__init__(*args, **kwargs)

    # Method for measuring dependent variable.
    @abstractmethod
    def manipulate(self):
        pass


class DV(Variable):
    def __init__(self, *args, **kwargs):
        self._name = "DV"
        self._variable_label = "Dependent Variable"

        self._is_covariate = False

        super().__init__(*args, **kwargs)

    # Method for measuring dependent variable.
    @abstractmethod
    def measure(self):
        pass

    # Get whether this dependent variable is treated as covariate.
    def __is_covariate__(self):
        return self._is_covariate

    # Set whether this dependent variable is treated as covariate.
    def __set_covariate__(self, is_covariate):
        self._is_covariate = is_covariate


class IVInSilico(IV):
    _variable_label = "IV"
    _name = "independent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    def __init__(self, *args, **kwargs):
        super(IVInSilico, self).__init__(*args, **kwargs)

    def manipulate(self):
        self._participant.set_value(self._name, self.get_value())


class DVInSilico(DV):
    _variable_label = "DV"
    _name = "dependent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    def __init__(self, *args, **kwargs):
        super(DVInSilico, self).__init__(*args, **kwargs)

    def measure(self):
        measurement = self._participant.get_value(self._name)
        self.set_value(measurement)


class IVTrial(IV):

    _name = "trial"
    _UID = ""
    _variable_label = "Trial"
    _units = "trials"
    _priority = 0
    _value_range = (0, 10000000)
    _value = 0

    def __init__(self, *args, **kwargs):
        super(IVTrial, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        pass

    def disconnect(self):
        pass


dv_labels = {}
iv_labels = {}


def register_iv_label(**kwargs):
    iv_labels.update(kwargs)
    return


def register_dv_label(**kwargs):
    dv_labels.update(kwargs)
    return


register_dv_label(
    **{
        "verbal_red": (
            DVInSilico,
            "Verbal Response Red",
            None,
            "verbal_red",
            "activation",
            0,
            (0, 1),
        ),
        "verbal_green": (
            DVInSilico,
            "Verbal Response Green",
            None,
            "verbal_green",
            "activation",
            0,
            (0, 1),
        ),
        "verbal_sample": (
            DVInSilico,
            "Verbal Response Sample",
            None,
            "verbal_sample",
            "class",
            0,
            (0, 1),
        ),
        "difference_detected": (
            DVInSilico,
            "Difference Detected",
            None,
            "difference_detected",
            "activation",
            0,
            (0, 1),
        ),
        "difference_detected_sample": (
            DVInSilico,
            "Difference Detected",
            None,
            "difference_detected_sample",
            "class",
            0,
            (0, 1),
        ),
        "learning_performance": (
            DVInSilico,
            "Accuracy",
            None,
            "learning_performance",
            "probability",
            0,
            (0, 1),
        ),
        "learning_performance_sample": (
            DVInSilico,
            "Accuracy Sample",
            None,
            "learning_performance_sample",
            "class",
            0,
            (0, 1),
        ),
        "dx1_lca": (
            DVInSilico,
            "dx1",
            None,
            "dx1_lca",
            "net input delta",
            0,
            (-1000, 1000),
        ),
    }
)

register_iv_label(
    **{
        "trial": (IVTrial, "Trial", "", "trial", "trials", 0, (0, 10000000)),
        "color_red": (
            IVInSilico,
            "Color Unit Red",
            None,
            "color_red",
            "activation",
            0,
            (0, 1),
        ),
        "color_green": (
            IVInSilico,
            "Color Unit Green",
            None,
            "color_green",
            "activation",
            0,
            (0, 1),
        ),
        "word_red": (
            IVInSilico,
            "Word Unit Red",
            None,
            "word_red",
            "activation",
            0,
            (0, 1),
        ),
        "word_green": (
            IVInSilico,
            "Word Unit Green",
            None,
            "word_green",
            "activation",
            0,
            (0, 1),
        ),
        "task_color": (
            IVInSilico,
            "Task Unit Color Naming",
            None,
            "task_color",
            "activation",
            0,
            (0, 1),
        ),
        "task_word": (
            IVInSilico,
            "Task Unit Word Reading",
            None,
            "task_word",
            "activation",
            0,
            (0, 1),
        ),
        "S1": (IVInSilico, "Stimulus 1 Intensity", None, "S1", "activation", 0, (0, 5)),
        "S2": (IVInSilico, "Stimulus 2 Intensity", None, "S2", "activation", 0, (0, 5)),
        "learning_trial": (
            IVInSilico,
            "Trial",
            None,
            "learning_trial",
            "trial",
            0,
            (0, 1000),
        ),
        "P_initial": (
            IVInSilico,
            "Initial Performance",
            None,
            "P_initial",
            "probability",
            0,
            (0, 1),
        ),
        "P_asymptotic": (
            IVInSilico,
            "Best Performance",
            None,
            "P_asymptotic",
            "probability",
            0,
            (0, 1),
        ),
        "x1_lca": (IVInSilico, "x1", None, "x1_lca", "net input", 0, (-1000, 1000)),
        "x2_lca": (IVInSilico, "x2", None, "x2_lca", "net input", 0, (-1000, 1000)),
        "x3_lca": (IVInSilico, "x3", None, "x3_lca", "net input", 0, (-1000, 1000)),
    }
)
