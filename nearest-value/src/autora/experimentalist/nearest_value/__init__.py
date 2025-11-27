from typing import Iterable, Union

import numpy as np
import pandas as pd

from autora.utils.deprecation import deprecated_alias


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: int,
):
    """
    A experimentalist which returns the nearest values between the input samples and the allowed
    values, without replacement.

    Args:
        conditions: The candidate samples of experimental conditions to be evaluated.
        reference_conditions: Experimental conditions to which the distance is calculated
        num_samples: number of samples

    Returns:
        the nearest values from `allowed_samples` to the `samples`

    """

    if isinstance(conditions, Iterable):
        conditions = np.array(list(conditions))

    if len(conditions.shape) == 1:
        conditions = conditions.reshape(-1, 1)

    if conditions.shape[0] < num_samples:
        raise Exception(
            "More samples requested than samples available in the set allowed of values."
        )

    X = np.array(reference_conditions)

    if X.shape[0] < num_samples:
        raise Exception("More samples requested than samples available in the pool.")

    x_new = np.empty((num_samples, conditions.shape[1]))

    # get index of row in x that is closest to each sample
    for row, sample in enumerate(X):

        if row >= num_samples:
            break

        dist = np.linalg.norm(conditions - sample, axis=1)
        idx = np.argmin(dist)
        x_new[row, :] = conditions[idx, :]
        conditions = np.delete(conditions, idx, axis=0)

    if isinstance(reference_conditions, pd.DataFrame):
        x_new = pd.DataFrame(x_new, columns=reference_conditions.columns)
    else:
        x_new = pd.DataFrame(x_new)

    return x_new


nearest_values_sample = sample
nearest_values_sampler = deprecated_alias(
    nearest_values_sample, "nearest_values_sampler"
)
