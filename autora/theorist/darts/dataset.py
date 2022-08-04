import numpy as np
import torch
from torch.utils.data import Dataset


class DARTSDataset(Dataset):
    def __init__(self, input_data: torch.tensor, output_data: torch.tensor):
        assert input_data.shape[0] == output_data.shape[0]
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self, experiment_id=None):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_tensor = self.input_data[idx]
        output_tensor = self.output_data[idx]
        return input_tensor, output_tensor


def darts_dataset_from_ndarray(input_data: np.ndarray, output_data: np.ndarray):

    obj = DARTSDataset(
        torch.tensor(input_data, dtype=torch.get_default_dtype()),
        torch.tensor(output_data, dtype=torch.get_default_dtype()),
    )
    return obj
