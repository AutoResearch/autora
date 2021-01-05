from abc import ABC, abstractmethod

class Participant_In_Silico:

    def __init__(self):
        pass

    # read value from participant
    @abstractmethod
    def get_value(self, variable_label):
        pass

    # assign value to participant
    @abstractmethod
    def set_value(self, variable_label, value):
        pass
