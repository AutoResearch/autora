from abc import ABC, abstractmethod
import numpy as np
from AER_experimentalist.experiment_environment.variable import Variable

class Tinkerforge_Variable(Variable):

    _variable_label = ""
    _UID = ""
    _priority = 0

    def __init__(self, variable_label="", UID="", name="", units="", priority="", value_range=(0,1), type=float):

        super().__init__(name=name, value_range=value_range, units=units, type=type, variable_label=variable_label)

        self._UID = UID
        self._priority = priority


    # Get priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __get_priority__(self):
        return self._priority

    # Set priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __set_priority__(self, priority):
        self._priority = priority

    def clean_up(self):
        pass