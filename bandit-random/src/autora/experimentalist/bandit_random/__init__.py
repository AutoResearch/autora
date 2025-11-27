"""
Experimentalist that returns
probability sequences: Sequences of vectors with elements between 0 and 1
or
reward sequences: Sequences of vectors with binary elements
"""

import numpy as np

from typing import Union, List, Optional
from collections.abc import Iterable


def pool_proba(
        num_probabilities: int,
        sequence_length: int,
        initial_probabilities: Optional[Iterable[Union[float, Iterable]]] = None,
        sigmas: Optional[Iterable[Union[float, Iterable]]] = None,
        num_samples: int = 1,
        random_state: Optional[int] = None,
) -> List[List[List[float]]]:
    """
    Returns a list of probability sequences.
    A probability sequence is a sequence of vectors of dimension `num_probabilities`. Each entry
    of this vector is a number between 0 and 1.
    We can set a fixed initial value for the first vector of each sequence and a constant drif rate.
    We can also set a range to randomly sample these values.


    Args:
        num_probabilities: The number of probilities/ dimention of each element of the sequence
        sequence_length: The length of the sequence
        initial_probabilities: A list of initial values for each element of the probalities. Each
        entry can be a range.
        sigmas: A list of sigma of the normal distribution for the drift rate of each arm. Each
            entry can be a range to be sampled from. The drift rate is defined as change per step
        num_samples: number of experimental conditions to select
        random_state: the seed value for the random number generator
    Returns:
        Sampled pool of experimental conditions

    Examples:
        We create a reward probabilty sequence for five two arm bandit tasks. The reward
        probabilities for each arm should be .5 and constant.
        >>> pool_proba(num_probabilities=2, sequence_length=3, num_samples=1, random_state=42)
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]

        If we want more arms:
        >>> pool_proba(num_probabilities=4, sequence_length=3, num_samples=1, random_state=42)
        [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]]

        longer sequence:
        >>> pool_proba(num_probabilities=2, sequence_length=5, num_samples=1, random_state=42)
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]

        more sequences:
        >>> pool_proba(num_probabilities=2, sequence_length=3, num_samples=2, random_state=42)
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]

        We  can set fixed initial values:
        >>> pool_proba(num_probabilities=2, sequence_length=3,
        ...     initial_probabilities=[0.,.4], random_state=42)
        [[[0.0, 0.4], [0.0, 0.4], [0.0, 0.4]]]

        And drift rates:
        >>> pool_proba(num_probabilities=2, sequence_length=3,
        ...     initial_probabilities=[0.,.4],
        ...     sigmas=[.1, .5], random_state=42)
        [[[0.0, 0.4], [0.030471707975443137, 0.7752255979032286], [0.0, 1.0]]]
        
        We can also sample the initial values by passing a range:
        >>> pool_proba(num_probabilities=2, sequence_length=3,
        ...     initial_probabilities=[[0, .2],[.8, 1.]],
        ...     sigmas=[[0., .25], [0., .5]],
        ...     random_state=42)
        [[[0.15479120971119267, 0.81883546957753], \
[0.23713042219259264, 0.8811974469636589], \
[0.34032881599649456, 0.7269307761486841]]]
    """
    rng = np.random.default_rng(random_state)
    if initial_probabilities:
        assert len(initial_probabilities) == num_probabilities
    else:
        initial_probabilities = [.5 for _ in range(num_probabilities)]
    if sigmas:
        assert len(sigmas) == num_probabilities
    else:
        sigmas = [0 for _ in range(num_probabilities)]
    res = []
    for _ in range(num_samples):
        seq = []
        for idx, el in enumerate(initial_probabilities):

            if _is_iterable(el):
                start = rng.uniform(el[0], el[1])
            else:
                start = el
            if _is_iterable(sigmas[idx]):
                sigma = rng.uniform(sigmas[idx][0], sigmas[idx][1])
            else:
                sigma = sigmas[idx]
            prob = [start]
            for _ in range(sequence_length - 1):
                start += rng.normal(loc=0, scale=sigma)
                start = max(0., min(start, 1.))
                prob.append(start)
            seq.append(prob)
        res.append(seq)
    for idx in range(len(res)):
        res[idx] = _transpose_matrix(res[idx])
    return res


def pool_from_proba(
        probability_sequence: Iterable,
        random_state: Optional[int] = None,
) -> List[List[List[float]]]:
    """
    From a given probability sequence sample rewards (0 or 1)

    Example:
        >>> proba_sequence = pool_proba(num_probabilities=2, sequence_length=3,
        ...     initial_probabilities=[.2,.8],
        ...     sigmas=[.2, .1], random_state=42)
        >>> proba_sequence
        [[[0.2, 0.8], [0.26094341595088627, 0.8750451195806458], \
[0.05294659470278715, 0.9691015912197671]]]
        >>> pool_from_proba(proba_sequence, 42)
        [[[0, 1], [1, 1], [0, 1]]]
    """
    rng = np.random.default_rng(random_state)
    probability_sequence_array = _sample_from_probabilities(probability_sequence, rng)
    probability_sequence_lst = [el for el in probability_sequence_array]
    return probability_sequence_lst


def pool(
        num_rewards: int,
        sequence_length: int,
        initial_probabilities: Optional[Iterable[Union[float, Iterable]]] = None,
        sigmas: Optional[Iterable[Union[float, Iterable]]] = None,
        num_samples: int = 1,
        random_state: Optional[int] = None,
) -> List[List[List[float]]]:
    """
    Returns a list of rewards.
    A reward sequence is a sequence of vectors of dimension `num_probabilities`. Each entry
    of this vector is a number between 0 and 1.
    We can set a fixed initial value for the reward probability of the first vector of each sequence
    and a constant drif rate.
    We can also set a range to randomly sample these values.


    Args:
        num_rewards: The number of rewards/ dimention of each element of the sequence
        sequence_length: The length of the sequence
        initial_probabilities: A list of initial reward-probabilities. Each
        entry can be a range.
        sigmas: A list of constant drift rate for each element of the probabilites. Each
        entry can be a range. The drift rate is defined as change per step
        num_samples: number of experimental conditions to select
        random_state: the seed value for the random number generator
    Returns:
        Sampled pool of experimental conditions

    Examples:
        We create a reward sequence for five two arm bandit tasks. The reward
        probabilities for each arm should be .5 and constant.
        >>> pool(num_rewards=2, sequence_length=3, num_samples=1, random_state=42)
        [[[1, 0], [1, 1], [0, 1]]]

        If we want more arms:
        >>> pool(num_rewards=4, sequence_length=3, num_samples=1, random_state=42)
        [[[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]]

        longer sequence:
        >>> pool(num_rewards=2, sequence_length=5, num_samples=1, random_state=42)
        [[[1, 0], [1, 1], [0, 1], [1, 1], [0, 0]]]

        more sequences:
        >>> pool(num_rewards=2, sequence_length=3, num_samples=2, random_state=42)
        [[[1, 0], [1, 1], [0, 1]], [[1, 1], [0, 0], [0, 1]]]

        We  can set fixed initial values:
        >>> pool(num_rewards=2, sequence_length=3,
        ...     initial_probabilities=[0.,.4],
        ...     random_state=42)
        [[[0, 0], [0, 1], [0, 1]]]

        And drift rates:
        >>> pool(num_rewards=2, sequence_length=3,
        ...     initial_probabilities=[0.,.4],
        ...     sigmas=[.2, .3],
        ...     random_state=42)
        [[[0, 0], [0, 1], [0, 1]]]

        We can also sample the initial values by passing a range:
        >>> pool(num_rewards=2, sequence_length=3,
        ...     initial_probabilities=[[0, .2],[.8, 1.]],
        ...     sigmas=[[0., .2], [0., .3]],
        ...     random_state=42)
        [[[0, 1], [1, 1], [0, 1]]]
    """
    _sequence = pool_proba(num_rewards,
                           sequence_length,
                           initial_probabilities,
                           sigmas,
                           num_samples,
                           random_state)
    return pool_from_proba(_sequence, random_state)


bandit_random_pool_proba = pool_proba
bandit_random_pool_from_proba = pool_from_proba
bandit_random_pool = pool


# Helper functions

def _sample_from_probabilities(prob_list, rng):
    """
    Helper function to sample values from a probability sequence
    """

    def sample_element(prob):
        return int(rng.choice([0, 1], p=[1 - prob, prob]))

    def recursive_sample(nested_list):
        if isinstance(nested_list, list):
            return [recursive_sample(sublist) for sublist in nested_list]
        else:
            return sample_element(nested_list)

    return np.array(recursive_sample(prob_list))


def _is_iterable(obj):
    """
    Helper function that returns true if an object is iterable
    """
    return isinstance(obj, Iterable)


def _transpose_matrix(matrix):
    """
    Helper function to transpose a list of lists.
    """
    return [list(row) for row in zip(*matrix)]
