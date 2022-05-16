from abc import ABC, abstractmethod
from tinkerforge_variable import Tinkerforge_Variable

class DV(Tinkerforge_Variable):

    def __init__(self, *args, **kwargs):
        self._name = "DV"
        self._variable_label = "Dependent Variable"

        self._is_covariate = False

        super(DV, self).__init__(*args, **kwargs)

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