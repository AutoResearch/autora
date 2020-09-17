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

    # Reads and sets value of independent variable from a dictionary with variable_label being the key
    def set_value_from_dict(self, dictionary, position):

        value_list  = dictionary.get(self.get_name())

        if value_list is None:
            raise Exception("Could not find value with name '" + self.get_name() + "' in dictionary.")

        if position > len(value_list):
            raise Exception("Queried position " + str(position) + " for variable " + self.get_name() + "'exceeds number of available positions for that variable in the dictionary.")

        self.set_value(value_list[position])

    # Set whether this dependent variable is treated as covariate.
    def set_covariate(self, is_covariate):
        self._is_covariate = is_covariate