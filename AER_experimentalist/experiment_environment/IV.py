from abc import ABC, abstractmethod
from tinkerforge_variable import Tinkerforge_Variable

class IV(Tinkerforge_Variable):

    _name = "IV"
    _variable_label = "Independent Variable"

    def __init__(self, *args, **kwargs):
        super(IV, self).__init__(*args, **kwargs)

    # Method for measuring dependent variable.
    @abstractmethod
    def manipulate(self):
        pass

    # Method for cleaning up measurement device.
    @abstractmethod
    def __clean_up__(self):
        pass
