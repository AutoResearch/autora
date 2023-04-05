from typing import Iterable, Sequence, Union

import numpy as np


def nearest_values_sampler(
    samples: Union[Iterable, Sequence],
    allowed_values: np.ndarray,
    n: int,
):
    """
    A sampler which returns the nearest values between the input samples and the allowed values,
    without replacement.

    Args:
        samples: input conditions
        allowed_samples: allowed conditions to sample from

    Returns:
        the nearest values from `allowed_samples` to the `samples`

    """

    if isinstance(allowed_values, Iterable):
        allowed_values = np.array(list(allowed_values))

    if len(allowed_values.shape) == 1:
        allowed_values = allowed_values.reshape(-1, 1)

    if isinstance(samples, Iterable):
        samples = np.array(list(samples))

    if allowed_values.shape[0] < n:
        raise Exception(
            "More samples requested than samples available in the set allowed of values."
        )

    if isinstance(samples, Iterable) or isinstance(samples, Sequence):
        samples = np.array(list(samples))

    if hasattr(samples, "shape"):
        if samples.shape[0] < n:
            raise Exception(
                "More samples requested than samples available in the pool."
            )

    x_new = np.empty((n, allowed_values.shape[1]))

    # get index of row in x that is closest to each sample
    for row, sample in enumerate(samples):

        if row >= n:
            break

        dist = np.linalg.norm(allowed_values - sample, axis=1)
        idx = np.argmin(dist)
        x_new[row, :] = allowed_values[idx, :]
        allowed_values = np.delete(allowed_values, idx, axis=0)

    return x_new
