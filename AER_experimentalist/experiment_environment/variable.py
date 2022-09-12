from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class outputTypes(Enum):
    REAL = 1
    SIGMOID = 2
    PROBABILITY = 3                 # single probability
    PROBABILITY_SAMPLE = 4          # sample from single probability
    PROBABILITY_DISTRIBUTION = 5    # probability distribution over classes
    CLASS = 6                       # sample from probability distribution over classes


class Variable():

    def __init__(self, name="", value_range=(0,1), units="", type = outputTypes.REAL, variable_label="", rescale=1):

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

    # Get range of variable.
    # The variable range determines the minimum and maximum allowed value to be manipulated or measured.
    def __get_value_range__(self):
        return self._value_range

    # Set range of variable.
    # The variable range determines the minimum and maximum allowed value to be manipulated or measured.
    def __set_value_range__(self, value_range):
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

    # Reads and sets value of independent variable from a dictionary with variable_label being the key
    def get_value_from_dict(self, dictionary, position):

        value_list  = dictionary.get(self.get_name()) # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception("Could not find value with name '" + self.get_name() + "' in dictionary.")

        if position > len(value_list):
            raise Exception("Queried position " + str(position) + " for variable " + self.get_name() + "'exceeds number of available positions for that variable in the dictionary.")

        return value_list[position] * self._rescale

    def get_value_list_from_dict(self, dictionary):
        value_list = dictionary.get(self.get_name())  # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception("Could not find value with name '" + self.get_name() + "' in dictionary.")

        rescaled_list = [element * self._rescale for element in value_list]

        return rescaled_list


    # Reads and sets value of independent variable from a dictionary with variable_label being the key
    def set_value_from_dict(self, dictionary, position):

        value_list  = dictionary.get(self.get_name())

        if value_list is None:
            raise Exception("Could not find value with name '" + self.get_name() + "' in dictionary.")

        if position > len(value_list):
            raise Exception("Queried position " + str(position) + " for variable " + self.get_name() + "'exceeds number of available positions for that variable in the dictionary.")

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
        self._unitt = units

    # Get variable label.
    def get_variable_label(self):
        return self._variable_label

    # Set variable label.
    def set_variable_label(self, variable_label):
        self._variable_label = variable_label