from abc import ABC
from AER_experimentalist.experiment_design import Experiment_Design
import numpy as np

class Experiment_Design_Synthetic_Weber(Experiment_Design, ABC):

    _stimulus_resolution = 1

    def __init__(self, stimulus_resolution):
        super(Experiment_Design_Synthetic_Weber, self).__init__()
        self._stimulus_resolution = stimulus_resolution

    def generate(self, object_of_study):

        experiment = dict()
        IV1 = object_of_study.independent_variables[0]
        IV2  = object_of_study.independent_variables[1]

        S1_levels = np.linspace(IV1._value_range[0], IV1._value_range[1], self._stimulus_resolution).tolist()
        S2_levels = np.linspace(IV1._value_range[0], IV1._value_range[1], self._stimulus_resolution).tolist()

        S1_trials = list()
        S2_trials = list()

        for S1 in S1_levels:
            for S2 in S2_levels:
                if S1 <= S2:
                    S1_trials.append(S1)
                    S2_trials.append(S2)

        experiment[IV1.get_name()] = S1_trials
        experiment[IV2.get_name()] = S2_trials

        return experiment

    def validate_trial(self, object_of_study, experiment_condition, experiment_sequence):

        IV1 = object_of_study.independent_variables[0]
        IV2 = object_of_study.independent_variables[1]

        S1 = experiment_condition[IV1.get_name()]
        S2 = experiment_condition[IV2.get_name()]

        if S1 <= S2:
            return True
        else:
            return False


