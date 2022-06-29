import copy
import random
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import aer.config as AER_cfg
from aer.variable import Variable, VariableCollection


class ObjectOfStudy(Dataset):
    """Collection of data."""

    key_experiment_id = "AER_Experiment"

    def __init__(
        self,
        name: str,
        independent_variables: Sequence[Variable],
        dependent_variables: Sequence[Variable],
        covariates: Sequence[Variable] = [],
    ):
        self._metadata: VariableCollection = VariableCollection(
            independent_variables=independent_variables,
            dependent_variables=dependent_variables,
            covariates=covariates,
        )

        self._name = name

        self.data: Dict[Any, Any] = dict()
        self._normalize_input = False
        self._normalize_output = False

        assert all(
            dv.type == dependent_variables[0].type for dv in dependent_variables
        ), (
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

    @property
    def metadata(self):
        return self._metadata

    @property
    def independent_variables(self):
        """The independent variables of the dataset."""
        return self.metadata.independent_variables

    @property
    def dependent_variables(self):
        """The dependent variables of the dataset."""
        return self.metadata.dependent_variables

    @property
    def covariates(self):
        """The covariates of the dataset."""
        return self.metadata.covariates

    @property
    def input_dimensions(self):
        """The number of independent variables and covariates."""
        return len(self.independent_variables) + len(self.covariates)

    @property
    def output_dimensions(self):
        """The number of dependent variables."""
        return len(self.dependent_variables)

    @property
    def output_type(self):
        """The ValueType of the first dependent variable."""
        return self.dependent_variables[0].type

    @property
    def name(self):
        """The name of the ObjectOfStudy."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

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
            input = self._normalize_variables(input, self.independent_variables)
        if self._normalize_output:
            output = self._normalize_variables(output, self.dependent_variables)

        return input, output

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
        input_dataset = torch.empty(num_data_points, self.input_length).float()
        output_dataset = torch.empty(num_data_points, self.output_length).float()

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

        dv_idx = self._get_dv_idx(dv)

        if iv2 is None:
            iv1_idx = self._get_iv_idx(iv1)
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
            iv1_idx = self._get_iv_idx(iv1)
            iv2_idx = self._get_iv_idx(iv2)
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

        # combine each IV with each DV
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

    def get_ivs_from_input(self, input, ivs):
        columns = list()
        if isinstance(ivs, list):
            for iv in ivs:
                if iv is not None:
                    columns.append(self._get_iv_idx(iv))
        else:
            columns.append(self._get_iv_idx(ivs))
        return input[:, columns]

    def get_dv_from_output(self, output, dv):
        column = self._get_dv_idx(dv)
        return output[:, column]

    def _get_iv_idx(self, iv):
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

    def _get_dv_idx(self, dv):
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

    def get_iv_limits_from_name(self, iv_name):

        for var in self.independent_variables:
            if var.get_name() == iv_name:
                return self.get_variable_limits(var)

        for var in self.covariates:
            if var.get_name() == iv_name:
                return self.get_variable_limits(var)

        return None

    def _get_variable_summary_stats(self, variables):

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

    def _normalize_variables(self, tensor, variables):
        def normalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
            return tensor

        # collect means and stds
        [means, stds] = self._get_variable_summary_stats(variables)

        # return normalized data
        return normalize(tensor, means, stds)

    def _unnormalize_variables(self, tensor, variables):
        def unnormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor

        # collect means and stds
        [means, stds] = self._get_variable_summary_stats(variables)

        # return normalized data
        return unnormalize(tensor, means, stds)

    @property
    def input_labels(self):
        _input_labels = list()
        for var in self.independent_variables:
            _input_labels.append(var.get_variable_label())
        for var in self.covariates:
            _input_labels.append(var.get_variable_label())

        return _input_labels

    @property
    def input_length(self):
        input_data = list()
        for var in self.independent_variables:
            input_data.append(var.get_value_from_dict(self.data, 0))
        for var in self.covariates:
            input_data.append(var.get_value_from_dict(self.data, 0))
        return len(input_data)

    @property
    def output_length(self):
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

    def add_data(self, new_data: Dict):
        for key in self.data.keys():
            if key in new_data:
                for value in new_data[key]:
                    self.data[key].append(value)
            else:
                raise Exception(
                    "Could not find key '" + key + "' in the new data dictionary."
                )
