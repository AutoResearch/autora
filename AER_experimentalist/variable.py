from abc import ABC, abstractmethod
import numpy as np

class Variable():

    _variable_label = ""
    _UID = ""
    _name = ""
    _units = ""
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    def __init__(self, variable_label="", UID="", name="", units="", priority="", value_range=(0,1)):

        self._variable_label = variable_label
        self._UID = UID
        self._name = name
        self._units = units
        self._priority = priority
        self._value_range = value_range


    # Get priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __get_priority__(self):
        return self._priority

    # Set priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __set_priority__(self, priority):
        self._priority = priority

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
        minimum = np.ones(value.shape) * self._value_range[0]
        maximum = np.ones(value.shape) * self._value_range[1]
        return np.min(np.max(value, minimum), maximum)

    # Get value.
    def get_value(self):
        return self._value

    # Set value.
    def set_value(self, value):
        self._value = self.__cap_value__(value)

    # Get variable label.
    def get_variable_label(self):
        return self._variable_label

    # Set variable label.
    def set_variable_label(self, variable_label):
        self._variable_label = variable_label