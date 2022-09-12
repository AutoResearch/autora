import AER_config as AER_cfg
import numpy as np
import AER_experimentalist.experimentalist_config as exp_cfg
from torch import nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from AER_utils import Plot_Types

from sweetpea.primitives import Factor
from sweetpea import fully_cross_block, synthesize_trials_non_uniform

from abc import ABC, abstractmethod

from AER_experimentalist.experimentalist import Experimentalist

import random

class Experimentalist_Random_Sampling(Experimentalist, ABC):

    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None, seed_data_file="", experiment_design=None, ivs=None):
        super().__init__(study_name, experiment_server_host, experiment_server_port, seed_data_file, experiment_design, ivs)

        experiment_design = list()
        resolution = 5 # hard coded to match self._seed_parameters[0] in experimentalist.py
        for var in ivs:
            factor = Factor(var.get_name(),
                            np.linspace(var._value_range[0], var._value_range[1], resolution).tolist())
            experiment_design.append(factor)

        block = fully_cross_block(experiment_design, experiment_design, [])

        experiment_sequence = synthesize_trials_non_uniform(block, 1)[0]

        sample = []
        num_samples = len(experiment_sequence[list(experiment_sequence.keys())[0]])
        for i in range(num_samples):
            cond = []
            for key in experiment_sequence:
                cond.append(float(experiment_sequence[key][i]))
            sample.append(cond)

        self.input_data = torch.Tensor(sample)
        self.indices = list(range(num_samples))  
        self.keys = list(experiment_sequence.keys())
        self.random = self.indices.copy()

    def init_experiment_search(self, model, object_of_study):
        super().init_experiment_search(model, object_of_study)
        return

    def sample_experiment_condition(self, model, object_of_study, condition):
        if condition == 0:
            random.shuffle(self.random)
        
        condition_data = self.input_data[self.random[condition]]

        condition = {}
        for j in range(len(self.keys)):
            condition[self.keys[j]] = float(condition_data[j])

        return condition
