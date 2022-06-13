from enum import Enum

import numpy as np


class OutputTypes(Enum):
    REAL = 'real'
    SIGMOID = 'sigmoid'
    PROBABILITY = 'probability'  # single probability
    PROBABILITY_SAMPLE = 'probability_sample'  # sample from single probability
    PROBABILITY_DISTRIBUTION = 'probability_distribution'  # probability distribution over classes
    CLASS = 'class'  # sample from probability distribution over classes


class Variable:

    def __init__(
            self,
            name="",
            value_range=(0, 1),
            units="",
            type=OutputTypes.REAL,
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

    # Cap value of variable
    def __cap_value__(self, value):
        minimum = self._value_range[0]
        maximum = self._value_range[1]
        return np.min([np.max([value, minimum]), maximum])

    # Get value.
    def get_value(self):
        return self._value * self._rescale

    # Set value.
    def set_value(self, value):
        self._value = self.__cap_value__(value)

    def get_value_from_dict(self, dictionary, position):
        """Reads and sets value of independent variable from a dictionary with
        variable_label being the key"""

        value_list = dictionary.get(self.get_name())  # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception(
                f"Could not find value with name '"
                f"{self.get_name()}"
                f" in dictionary."
            )

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
        value_list = dictionary.get(self.get_name())  # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception(
                f"Could not find value with name '"
                f"{self.get_name()}"
                f"' in dictionary."
            )

        rescaled_list = [element * self._rescale for element in value_list]

        return rescaled_list

    # Reads and sets value of independent variable from a dictionary
    # with variable_label being the key
    def set_value_from_dict(self, dictionary, position):

        value_list = dictionary.get(self.get_name())

        if value_list is None:
            raise Exception(
                f"Could not find value with name '"
                f"{self.get_name()}"
                f"' in dictionary."
            )

        if position > len(value_list):
            raise Exception(
                f"Queried position "
                f"{str(position)}"
                f" for variable "
                f"{self.get_name()}"
                f" exceeds number of available positions for that variable in the dictionary."
            )

        self.set_value(value_list[position])

    # Get variable name.
    def get_name(self):
        return self._name

    # Set variable name.
    def set_name(self, name):
        self._name = name

    # Get variable units.
    def get_units(self):
        return self._units

    # Set variable units.
    def set_units(self, units):
        self._units = units

    # Get variable label.
    def get_variable_label(self):
        return self._variable_label

    # Set variable label.
    def set_variable_label(self, variable_label):
        self._variable_label = variable_label

    def set_covariate(self, is_covariate):
        """Set whether this dependent variable is treated as covariate."""
        self._is_covariate = is_covariate


class IVInSilico(Variable):
    _variable_label = "IV"
    _name = "independent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):
        super(IVInSilico, self).__init__(*args, **kwargs)

    def manipulate(self):
        self._participant.set_value(self._name, self.get_value())


class DVInSilico(Variable):
    _variable_label = "DV"
    _name = "dependent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):
        super(DVInSilico, self).__init__(*args, **kwargs)

    def measure(self):
        measurement = self._participant.get_value(self._name)
        self.set_value(measurement)


