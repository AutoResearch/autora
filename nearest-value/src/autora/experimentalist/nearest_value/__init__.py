from typing import Iterable, Union

import numpy as np
import pandas as pd

from autora.utils.deprecation import deprecated_alias


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    allowed_values: np.ndarray,
    num_samples: int,
):
    """
    A experimentalist which returns the nearest values between the input samples and the allowed
    values, without replacement.

    Args:
        conditions: input conditions
        allowed_values: allowed conditions to sample from
        num_samples: number of samples

    Returns:
        the nearest values from `allowed_samples` to the `samples`

    """

    if isinstance(allowed_values, Iterable):
        allowed_values = np.array(list(allowed_values))

    if len(allowed_values.shape) == 1:
        allowed_values = allowed_values.reshape(-1, 1)

    if allowed_values.shape[0] < num_samples:
        raise Exception(
            "More samples requested than samples available in the set allowed of values."
        )

    X = np.array(conditions)

    if X.shape[0] < num_samples:
        raise Exception("More samples requested than samples available in the pool.")

    x_new = np.empty((num_samples, allowed_values.shape[1]))

    # get index of row in x that is closest to each sample
    for row, sample in enumerate(X):

        if row >= num_samples:
            break

        dist = np.linalg.norm(allowed_values - sample, axis=1)
        idx = np.argmin(dist)
        x_new[row, :] = allowed_values[idx, :]
        allowed_values = np.delete(allowed_values, idx, axis=0)

    if isinstance(conditions, pd.DataFrame):
        x_new = pd.DataFrame(x_new, columns=conditions.columns)

    return x_new


nearest_values_sample = sample
nearest_values_sampler = deprecated_alias(
    nearest_values_sample, "nearest_values_sampler"
)
