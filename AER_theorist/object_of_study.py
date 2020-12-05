from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from enum import Enum
from AER_experimentalist.experiment_environment.variable import *
import AER_config as AER_cfg
from typing import List, Dict
import torch
import numpy as np

class Object_Of_Study(Dataset):

    independent_variables = list()
    dependent_variables = list()
    covariates = list()
    data = dict()

    _experiment_label = 'AER_experiment'
    _normalize_input = False
    _normalize_output = False

    def __init__(self, name, independent_variables: List[Variable], dependent_variables: List[Variable], covariates=list(), input_dimensions=None, output_dimensions=None, output_type=None):

        self.name = name

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
            self.data[var.get_name()] = list()
        for var in self.independent_variables:
            self.data[var.get_name()] = list()
        for var in self.covariates:
            self.data[var.get_name()] = list()
        self.data[AER_cfg.experiment_label] = list()

    def __len__(self):
        return len(self.data[AER_cfg.experiment_label])

    def __getitem__(self, idx):

        # get input data
        input_data = list()
        for var in self.independent_variables:
            input_data.append(var.get_value_from_dict(self.data, idx))
        for var in self.covariates:
            input_data.append(var.get_value_from_dict(self.data, idx))

        # get output data
        output_data = list()
        for var in self.dependent_variables:
            output_data.append(var.get_value_from_dict(self.data, idx))

        input = torch.tensor(input_data).float()
        output = torch.tensor(output_data).float()

        # normalize if required
        if self._normalize_input:
            input = self.normalize_variables(input, self.independent_variables)
        if self._normalize_output:
            output = self.normalize_variables(output, self.dependent_variables)

        return input, output

    # potentially redundant with: get_all_data
    def get_dataset(self):

        # create empty tensor
        input_dataset = torch.empty(len(self), self.__get_input_length__()).float()
        output_dataset = torch.empty(len(self), self.__get_output_length__()).float()

        for idx in range(len(self)):
            (input, output) = self.__getitem__(idx)
            input_dataset[idx,:] = input
            output_dataset[idx, :] = output

        return input_dataset, output_dataset

    def get_counterbalanced_input(self, resolution):

        factor_levels = list()
        independent_variables = self.independent_variables + self.covariates
        for var in independent_variables:
            var_levels = np.linspace(var.__get_value_range__()[0] * var._rescale,
                                     var.__get_value_range__()[1] * var._rescale,
                                     resolution)
            factor_levels.append(var_levels)

        input_np = np.array(np.meshgrid(*factor_levels)).T.reshape(-1,len(independent_variables))

        input = torch.tensor(input_np).float()

        return input

    def average_DV_for_IVs(self, DV, IVs, input, output):
        IV1 = IVs[0]
        IV2 = IVs[1]

        DV_idx = self.get_DV_idx(DV)

        if IV2 is None:
            IV1_idx = self.get_IV_idx(IV1)
            unique_IV_values = np.unique(input[:,IV1_idx])
            DV_values = np.empty(unique_IV_values.shape)
            for row, element in enumerate(unique_IV_values):
                value_log = list()
                for idx in range(output.shape[0]):
                    if element == input[idx, IV1_idx]:
                        value_log.append(output[idx, DV_idx])
                value_mean = np.mean(value_log)
                DV_values[row] = value_mean
        else:
            IV1_idx = self.get_IV_idx(IV1)
            IV2_idx = self.get_IV_idx(IV2)
            unique_IV_rows = np.unique(input[:, [IV1_idx, IV2_idx]])
            DV_values = np.empty((unique_IV_rows.shape[0], 1))
            for row, combination in enumerate(unique_IV_rows):
                value_log = list()
                for idx in range(output.shape[0]):
                    if combination == input[idx, [IV1_idx, IV2_idx]]:
                        value_log.append(output[idx, DV_idx])
                value_mean = np.mean(value_log)
                DV_values[row] = value_mean

        return DV_values

    def get_variable_data(self, variable):
        var_data = list()
        for idx in len(self):
            var_data.append(variable.get_value_from_dict(self.data, idx))
        IV_data = torch.tensor(var_data).float()
        return IV_data

    def get_IVs_from_input(self, input, IVs):
        columns = list()
        if isinstance(IVs, list):
            for IV in IVs:
                if IV is not None:
                    columns.append(self.get_IV_idx(IV))
        else:
            columns.append(self.get_IV_idx(IVs))
        return input[:, columns]

    def get_DV_from_output(self, output, DV):
        column = self.get_DV_idx(DV)
        return output[:, column]

    def get_IV_idx(self, IV):
        column = None
        for idx, var in enumerate(self.independent_variables):
            if var.get_name() == IV.get_name():
                column = idx
                break
        for idx, var in enumerate(self.covariates):
            if var.get_name() == IV.get_name():
                column = idx + len(self.independent_variables)
                break
        return column

    def get_DV_idx(self, DV):
        column = None
        for idx, var in enumerate(self.dependent_variables):
            if var.get_name() == DV.get_name():
                column = idx
                break
        return column

    def get_variable_limits(self, var):
        limits = [var.__get_value_range__()[0] * var._rescale, var.__get_value_range__()[1] * var._rescale]
        return limits

    def get_variable_summary_stats(self, variables):

        # collect means and stds
        means = list()
        stds = list()
        for var in variables:
            IV_data = self.data[var.get_name()][0]
            m = np.mean(IV_data)
            s = np.std(IV_data)
            means.append(m)
            stds.append(s)

        return [means, stds]

    def normalize_variables(self, tensor, variables):
        # collect means and stds
        [means, stds] = self.get_variable_summary_stats(variables)

        # return normalized data
        return normalize(tensor, means, stds)

    def unnormalize_variables(self, tensor, variables):
        # collect means and stds
        [means, stds] = self.get_variable_summary_stats(variables)

        # return normalized data
        return unnormalize(tensor, means, stds)

    def get_name(self):
        return self.name

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

    def __get_input_names__(self):
        input_names = list()
        for var in self.independent_variables:
            input_names.append(var.get_name())
        for var in self.covariates:
            input_names.append(var.get_name())

        return input_names

    def __get_input_length__(self):
        input_data = list()
        for var in self.independent_variables:
            input_data.append(var.get_value_from_dict(self.data, 0))
        for var in self.covariates:
            input_data.append(var.get_value_from_dict(self.data, 0))
        return len(input_data)

    def __get_output_length__(self):
        output_data = list()
        for var in self.dependent_variables:
            output_data.append(var.get_value_from_dict(self.data, 0))
        return len(output_data)

    # potentially redundant with: get_dataset
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
                for value in new_data[key]:
                    self.data[key].append(value)
            else:
                raise Exception("Could not find key '" + key + "' in the new data dictionary.")


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def unnormalize( tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor