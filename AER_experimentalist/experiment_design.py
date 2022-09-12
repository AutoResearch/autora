from abc import ABC, abstractmethod

class Experiment_Design(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, object_of_study):
        pass

    def validate_trial(self, object_of_study, experiment_condition, experiment_sequence):
        return True