import copy
import random
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import aer_config as AER_cfg
from aer.variable import Variable


class Object_Of_Study(Dataset):

    key_experiment_id = "AER_Experiment"

    def __init__(
        self,
        name,
        independent_variables: Sequence[Variable],
        dependent_variables: Sequence[Variable],
        covariates: Sequence[Variable] = None,
        input_dimensions=None,
        output_dimensions=None,
        output_type=None,
    ):

        self.name = name

        self.independent_variables: Sequence[Variable] = list()
        self.dependent_variables: Sequence[Variable] = list()
        self.covariates: Sequence[Variable] = list()
        self.data: Dict[Any, Any] = dict()
        self._normalize_input = False
        self._normalize_output = False

        # set independent and dependent variables
        self.independent_variables.extend(independent_variables)
        self.dependent_variables.extend(dependent_variables)

        if covariates is not None:
            self.covariates.extend(covariates)

        # set number of output dimensions
        if output_dimensions is None:
            self.output_dimensions = len(self.dependent_variables)
        else:
            self.output_dimensions = output_dimensions

        # set number of input dimensions
        if input_dimensions is None:
            self.input_dimensions = len(self.independent_variables) + len(
                self.covariates
            )
        else:
            self.input_dimensions = input_dimensions

        # set output type
        self.output_type = self.dependent_variables[0].type
        for variable in dependent_variables:
            if variable.type != self.output_type:
                AttributeError(
                    "Dependent variable output types don't match. "
                    "Different output types are not supported yet."
                )

        # set up data
        for dv in self.dependent_variables:
            self.data[dv.get_name()] = list()
        for iv in self.independent_variables:
            self.data[iv.get_name()] = list()
        for cv in self.covariates:
            self.data[cv.get_name()] = list()
        self.data[AER_cfg.experiment_label] = list()

    def __len__(self, experiment_id=None):
        if experiment_id is None:
            return len(self.data[self.key_experiment_id])
        else:
            return self.data[self.key_experiment_id].count(experiment_id)

    def __getitem__(self, idx, experiment_id=None):

        # determine relevant experiment id

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

    def get_random_input_sample(self):

        # sample input data
        input_data = list()

        for var in self.independent_variables:
            sample = np.random.uniform(
                var.__get_value_range__()[0] * var._rescale,
                var.__get_value_range__()[1] * var._rescale,
            )
            input_data.append(sample)

        for var in self.covariates:
            sample = np.random.uniform(
                var.__get_value_range__()[0] * var._rescale,
                var.__get_value_range__()[1] * var._rescale,
            )
            input_data.append(sample)

        input = torch.tensor(input_data).float()

        # normalize if required
        if self._normalize_input:
            input = self.normalize_variables(input, self.independent_variables)

        return input

    def new_experiment_sequence(self):
        experiment_sequence = dict()

        for var in self.independent_variables:
            experiment_sequence[var.get_name()] = list()

        for var in self.covariates:
            experiment_sequence[var.get_name()] = list()

        return experiment_sequence

    def get_last_experiment_id(self):
        return np.max(self.data[self.key_experiment_id])

    def get_experiment_indices(self, experiment_id):
        indices = [
            i
            for i, x in enumerate(self.data[self.key_experiment_id])
            if x == experiment_id
        ]
        return indices

    # potentially redundant with: get_all_data
    def get_dataset(self, experiment_id=None):

        # determine length of data set
        if experiment_id is None:
            num_data_points = len(self)
        else:
            num_data_points = self.__len__(experiment_id)

        # create an empty tensor
        input_dataset = torch.empty(
            num_data_points, self.__get_input_length__()
        ).float()
        output_dataset = torch.empty(
            num_data_points, self.__get_output_length__()
        ).float()

        if experiment_id is None:
            for idx in range(len(self)):
                (input, output) = self.__getitem__(idx)
                input_dataset[idx, :] = input
                output_dataset[idx, :] = output
        else:
            experiment_indices = self.get_experiment_indices(experiment_id)
            sub_idx = 0
            for idx in range(len(self)):
                (input, output) = self.__getitem__(idx)
                if idx in experiment_indices:
                    input_dataset[sub_idx, :] = input
                    output_dataset[sub_idx, :] = output
                    sub_idx += 1

        return input_dataset, output_dataset

    def get_counterbalanced_input(self, resolution):

        factor_levels = list()
        independent_variables = self.independent_variables + self.covariates
        for var in independent_variables:
            var_levels = np.linspace(
                var.__get_value_range__()[0] * var._rescale,
                var.__get_value_range__()[1] * var._rescale,
                resolution,
            )
            factor_levels.append(var_levels)

        input_np = np.array(np.meshgrid(*factor_levels)).T.reshape(
            -1, len(independent_variables)
        )

        input = torch.tensor(input_np).float()

        return input

    def average_dv_for_ivs(self, dv, ivs, input, output):
        iv1 = ivs[0]
        iv2 = ivs[1]

        dv_idx = self.get_dv_idx(dv)

        if iv2 is None:
            iv1_idx = self.get_iv_idx(iv1)
            unique_iv_values = np.unique(input[:, iv1_idx])
            dv_values = np.empty(unique_iv_values.shape)
            for row, element in enumerate(unique_iv_values):
                value_log = list()
                for idx in range(output.shape[0]):
                    if element == input[idx, iv1_idx]:
                        value_log.append(output[idx, dv_idx])
                value_mean = np.mean(value_log)
                dv_values[row] = value_mean
            return unique_iv_values, dv_values
        else:
            iv1_idx = self.get_iv_idx(iv1)
            iv2_idx = self.get_iv_idx(iv2)
            unique_iv_rows = np.unique(input[:, [iv1_idx, iv2_idx]], axis=0)
            dv_values = np.empty((unique_iv_rows.shape[0]))
            iv1_values = np.empty((unique_iv_rows.shape[0]))
            iv2_values = np.empty((unique_iv_rows.shape[0]))
            for row, combination in enumerate(unique_iv_rows):
                value_log = list()
                for idx in range(output.shape[0]):
                    if (combination == input[idx, [iv1_idx, iv2_idx]]).all():
                        value_log.append(output[idx, dv_idx])
                value_mean = np.mean(value_log)
                dv_values[row] = value_mean
                iv1_values[row] = combination[0]
                iv2_values[row] = combination[1]
            unique_iv_values = (iv1_values, iv2_values)
            return unique_iv_values, dv_values

    def get_plot_list(self):
        iv_list_1 = list()
        iv_list_2 = list()
        dv_list = list()

        # combine each IV with each IV with each DV
        independent_variables_1 = self.independent_variables + self.covariates
        independent_variables_2 = [None] + self.independent_variables + self.covariates

        for iv1 in independent_variables_1:
            for iv2 in independent_variables_2:
                for dv in self.dependent_variables:
                    if iv1 != iv2:
                        iv_list_1.append(iv1)
                        iv_list_2.append(iv2)
                        dv_list.append(dv)

        # combine each IV
        return (iv_list_1, iv_list_2, dv_list)

    def get_variable_data(self, variable):
        var_data = list()
        for idx in len(self):
            var_data.append(variable.get_value_from_dict(self.data, idx))
        iv_data = torch.tensor(var_data).float()
        return iv_data

    def get_ivs_from_input(self, input, ivs):
        columns = list()
        if isinstance(ivs, list):
            for iv in ivs:
                if iv is not None:
                    columns.append(self.get_iv_idx(iv))
        else:
            columns.append(self.get_iv_idx(ivs))
        return input[:, columns]

    def get_DV_from_output(self, output, dv):
        column = self.get_dv_idx(dv)
        return output[:, column]

    def get_iv_idx(self, iv):
        column = None
        for idx, var in enumerate(self.independent_variables):
            if var.get_name() == iv.get_name():
                column = idx
                break
        for idx, var in enumerate(self.covariates):
            if var.get_name() == iv.get_name():
                column = idx + len(self.independent_variables)
                break
        return column

    def get_dv_idx(self, dv):
        column = None
        for idx, var in enumerate(self.dependent_variables):
            if var.get_name() == dv.get_name():
                column = idx
                break
        return column

    def get_iv_name(self, idx):
        if idx < len(self.independent_variables):
            name = self.independent_variables[idx].get_name()
        else:
            idx = idx - len(self.independent_variables)
            if idx < len(self.covariates):
                name = self.covariates[idx].get_name()
            else:
                raise Exception("Index exceeds number of independent variables.")
        return name

    def get_dv_name(self, idx):
        if idx < len(self.dependent_variables):
            name = self.dependent_variables[idx].get_name()
            return name
        else:
            raise Exception("Index exceeds number of dependent variables.")

    def get_variable_limits(self, var):
        limits = [
            var.__get_value_range__()[0] * var._rescale,
            var.__get_value_range__()[1] * var._rescale,
        ]
        return limits

    def rescale_experiment_sequence(self, sequence):
        rescaled_sequence = dict()
        for key in sequence:
            values = sequence[key]
            rescale = self.get_iv_rescale_from_name(key)
            values_rescaled = [val * rescale for val in values]
            rescaled_sequence[key] = values_rescaled

        return rescaled_sequence

    def get_iv_rescale_from_name(self, iv_name):

        for var in self.independent_variables:
            if var.get_name() == iv_name:
                return var._rescale

        for var in self.covariates:
            if var.get_name() == iv_name:
                return var._rescale

    def get_IV_limits_from_name(self, iv_name):

        for var in self.independent_variables:
            if var.get_name() == iv_name:
                return self.get_variable_limits(var)

        for var in self.covariates:
            if var.get_name() == iv_name:
                return self.get_variable_limits(var)

        return None

    def get_variable_summary_stats(self, variables):

        # collect means and stds
        means = list()
        stds = list()
        for var in variables:
            iv_data = self.data[var.get_name()][0]
            m = np.mean(iv_data)
            s = np.std(iv_data)
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

    def split(self, proportion=0.5):

        split_copy = copy.deepcopy(self)

        # determine indices to be split
        num_data_points = self.__len__()
        indices = range(num_data_points)
        num_samples = round(proportion * num_data_points)
        samples = random.sample(indices, num_samples)

        split_copy.data = dict()

        # first add samples to the new copy
        for key in self.data.keys():
            split_copy.data[key] = list()
            for samp in samples:
                split_copy.data[key].append(self.data[key][samp])

        # now remove samples from original object
        for key in self.data.keys():
            values = self.data[key]
            values = [i for j, i in enumerate(values) if j not in samples]
            self.data[key] = values

        return split_copy

    # potentially redundant with: get_dataset
    def get_all_data(self):

        num_patterns = self.__len__()

        if num_patterns > 0:
            input_tensor, output_tensor = self.__getitem__(0)
        else:
            input_tensor = torch.Variable(
                np.empty((0, self.input_dimensions), dtype=np.float32)
            )
            output_tensor = torch.Variable(
                np.empty((0, self.output_dimensions), dtype=np.float32)
            )

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
                raise Exception(
                    "Could not find key '" + key + "' in the new data dictionary."
                )


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
