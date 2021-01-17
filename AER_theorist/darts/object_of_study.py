from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from enum import Enum
from AER_experimentalist.experiment_environment.variable import *
import AER_config as AER_cfg
from typing import List, Dict
import torch
import copy

class Object_Of_Study(Dataset):

    independent_variables = list()
    dependent_variables = list()
    covariates = list()
    data = dict()

    _experiment_label = 'AER_experiment'

    def __init__(self, independent_variables: List[Variable], dependent_variables: List[Variable], covariates=list(), input_dimensions=None, output_dimensions=None, output_type=None):

        # set independent and dependent variables
        if len(independent_variables) == 0:
            Exception("No independent variables specified.")

        if len(dependent_variables) == 0:
            Exception("No dependent variables specified.")

        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables
        self.covariates = covariates

        # set number of output dimensions
        if output_dimensions is None:
            self.output_dimensions = len(self.dependent_variables)
        else:
            self.output_dimensions = output_dimensions

        # set number of input dimensions
        if input_dimensions is None:
            self.input_dimensions = len(self.independent_variables) + len(self.covariates)
        else:
            self.input_dimensions = input_dimensions

        # set output type
        self.output_type = self.dependent_variables[0].type
        for variable in dependent_variables:
            if variable.type != self.output_type:
                Exception("Dependent variable output types don't match. Different output types are not supported yet.")

        # set up data
        for var in self.dependent_variables:
            self.data[var.get_variable_label()] = list()
        for var in self.independent_variables:
            self.data[var.get_variable_label()] = list()
        for var in self.covariates:
            self.data[var.get_variable_label()] = list()
        self.data[AER_cfg.experiment_label] = list()

        def __len__(self):
            return len(self.data[AER_cfg.experiment_label])

        def __getitem__(self, idx):

            # get input data
            input_data = list()
            for var in self.independent_variables:
                input_data.append(self.data[var.get_name()][idx])
            for var in self.covariates:
                input_data.append(self.data[var.get_name()][idx])

            # get output data
            output_data = list()
            for var in self.dependent_variables:
                output_data.append(self.data[var.get_name()][idx])

            output = torch.tensor(output_data)
            input = torch.tensor(input_data)

            return input, output

        def __get_input_dim__(self):
            return self.input_dimensions

        def __get_output_dim__(self):
            return self.output_dimensions

        def __get_output_type__(self):
            return self.output_type

        def __get_input_labels__(self):
            input_labels = list()
            for var in self.independent_variables:
                input_labels.append(var.get_variable_label())
            for var in self.covariates:
                input_labels.append(var.get_variable_label())

            return input_labels

        def get_all_data(self):

            num_patterns = self.__len__()

            if num_patterns > 0:
                input_tensor, output_tensor = self.__getitem__(0)
            else:
                input_tensor = torch.Variable(np.empty((0, self.input_dimensions), dtype=np.float32))
                output_tensor = torch.Variable(np.empty((0, self.output_dimensions), dtype=np.float32))

            for idx in range(1, num_patterns):
                tmp_input_tensor, tmp_output_tensor = self.__getitem__(idx)
                input_tensor = torch.cat((input_tensor, tmp_input_tensor), 0)
                output_tensor = torch.cat((output_tensor, tmp_output_tensor), 0)

        def add_data(self, new_data: Dict):
            for key in self.data.keys():
                if key in new_data:
                    self.data[key].append(new_data[key])
                else:
                    raise Exception("Could not find key '" + key + "' in the new data dictionary.")

class objectOfStudy(Dataset, ABC):

    def __init__(self, num_patterns=100, sampling=False):
        self._num_patterns = num_patterns
        self._sampling = sampling

    @abstractmethod
    def __get_input_dim__(self):
        pass

    @abstractmethod
    def __get_output_dim__(self):
        pass

    @abstractmethod
    def __get_output_type__(self):
        pass

    @abstractmethod
    def __get_input_labels__(self):
        pass

    @abstractmethod
    def __get_name__(self):
        pass

    @abstractmethod
    def sample_model_fit(self):
        pass

    @abstractmethod
    def sample_model_fit_2d(self):
        pass




