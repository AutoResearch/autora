from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DARTSDataset(Dataset):
    """
    A dataset for the DARTS algorithm.
    """

    def __init__(self, input_data: torch.tensor, output_data: torch.tensor):
        """
        Initializes the dataset.

        Arguments:
            input_data: The input data to the dataset.
            output_data: The output data to the dataset.
        """
        assert input_data.shape[0] == output_data.shape[0]
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self, experiment_id: Optional[int] = None) -> int:
        """
        Returns the length of the dataset.

        Arguments:
            experiment_id:

        Returns:
            The length of the dataset.
        """
        return self.input_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns the item at the given index.

        Arguments:
            idx: The index of the item to return.

        Returns:
            The item at the given index.

        """
        input_tensor = self.input_data[idx]
        output_tensor = self.output_data[idx]
        return input_tensor, output_tensor


def darts_dataset_from_ndarray(
    input_data: np.ndarray, output_data: np.ndarray
) -> DARTSDataset:
    """
    A function to create a dataset from numpy arrays.

    Arguments:
        input_data: The input data to the dataset.
        output_data: The output data to the dataset.

    Returns:
        The dataset.

    """

    obj = DARTSDataset(
        torch.tensor(input_data, dtype=torch.get_default_dtype()),
        torch.tensor(output_data, dtype=torch.get_default_dtype()),
    )
    return obj
