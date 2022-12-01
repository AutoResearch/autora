import random
from typing import Iterable, Sequence, Union


def random_sampler(conditions: Union[Iterable, Sequence], n: int):
    """
    Uniform random sampling without replacement from a pool of conditions.
    Args:
        conditions: Pool of conditions
        n: number of samples to collect

    Returns: Sampled pool

    """

    if isinstance(conditions, Iterable):
        conditions = list(conditions)
    random.shuffle(conditions)
    samples = conditions[0:n]

    return samples
