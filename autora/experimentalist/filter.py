from enum import Enum
from typing import Callable, Iterable, Tuple

import numpy as np


def weber_filter(values):
    return filter(lambda s: s[0] <= s[1], values)


def train_test_filter(
    seed: int = 180, train_p: float = 0.5
) -> Tuple[Callable[[Iterable], Iterable], Callable[[Iterable], Iterable]]:
    """
    A pipeline filter which pseudorandomly assigns values from the input into "train" or "test"
    groups. This is particularly useful when working with streams of data of potentially
    unbounded length.

    This isn't a great method for small datasets, as it doesn't guarantee producing training
    and test sets which are as close as possible to the specified desired proportions.
    Consider using the scikit-learn `train_test_split` for cases where it's practical to
    enumerate the full dataset in advance.

    Args:
        seed: random number generator seeding value
        train_p: proportion of data which go into the training set. A float between 0 and 1.

    Returns:
        a tuple of callables `(train_filter, test_filter)` which split the input data
            into two complementary streams.


    Examples:
        We can create complementary train and test filters using the function:
        >>> train_filter, test_filter = train_test_filter(train_p=0.6, seed=180)

        The `train_filter` generates a sequence of ~60% of the input list –
        in this case, 15 of 20 datapoints.
        Note that the correct split would be 12 of 20 data points.
        Again, for data with bounded length it is advisable
        to use scikit-learn `train_test_split` instead.
        >>> list(train_filter(range(20)))
        [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18, 19]

        When we run the `test_filter`, it fills in the gaps, giving us the remaining 5 values:
        >>> list(test_filter(range(20)))
        [1, 7, 8, 13, 14]

        We can continue to generate new values for as long as we like using the same filter and the
        continuation of the input range:
        >>> list(train_filter(range(20, 40)))
        [20, 22, 23, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39]

        ... and some more.
        >>> list(train_filter(range(40, 50)))
        [41, 42, 44, 45, 46, 49]

        As the number of samples grows, the fraction in the train and test sets
        will approach `train_p` and `1 - train_p`.

        The test_filter fills in the gaps again.
        >>> list(test_filter(range(20, 30)))
        [21, 24, 25, 26]

        If you rerun the *same* test_filter on a fresh range, then the results will be different
        to the first time around:
        >>> list(test_filter(range(20)))
        [5, 10, 13, 17, 18]

        ... but if you regenerate the test_filter, it'll reproduce the original sequence
        >>> _, test_filter_regenerated = train_test_filter(train_p=0.6, seed=180)
        >>> list(test_filter_regenerated(range(20)))
        [1, 7, 8, 13, 14]

        It also works on tuple-valued lists:
        >>> from itertools import product
        >>> train_filter_tuple, test_filter_tuple = train_test_filter(train_p=0.3, seed=42)
        >>> list(test_filter_tuple(product(["a", "b"], [1, 2, 3])))
        [('a', 1), ('a', 2), ('a', 3), ('b', 1), ('b', 3)]

        >>> list(train_filter_tuple(product(["a","b"], [1,2,3])))
        [('b', 2)]

        >>> from itertools import count, takewhile
        >>> train_filter_unbounded, test_filter_unbounded = train_test_filter(train_p=0.5, seed=21)

        >>> list(takewhile(lambda s: s < 90, count(79)))
        [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]

        >>> train_pool = train_filter_unbounded(count(79))
        >>> list(takewhile(lambda s: s < 90, train_pool))
        [82, 85, 86, 89]

        >>> test_pool = test_filter_unbounded(count(79))
        >>> list(takewhile(lambda s: s < 90, test_pool))
        [79, 80, 81, 83, 84, 87, 88]

        >>> list(takewhile(lambda s: s < 110, test_pool))
        [91, 93, 94, 97, 100, 105, 106, 109]

    """

    test_p = 1 - train_p

    _TrainTest = Enum("_TrainTest", ["train", "test"])

    def train_test_stream():
        """Generates a pseudorandom stream of _TrainTest.train and _TrainTest.test."""
        rng = np.random.default_rng(seed)
        while True:
            yield rng.choice([_TrainTest.train, _TrainTest.test], p=(train_p, test_p))

    def _factory(allow):
        """Factory to make complementary generators which split their input
        corresponding to the values of the pseudorandom train_test_stream."""
        _stream = train_test_stream()

        def _generator(values):
            """Generator which yields items from the `values` depending on
            whether the corresponding item from the `_stream`
            matches the `allow` parameter."""
            for v, train_test in zip(values, _stream):
                if train_test == allow:
                    yield v

        return _generator

    return _factory(_TrainTest.train), _factory(_TrainTest.test)
