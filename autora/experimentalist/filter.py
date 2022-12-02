from enum import Enum

import numpy as np


def weber_filter(values):
    return filter(lambda s: s[0] <= s[1], values)


def train_test_filter(seed=180, train_p=0.5):
    """
    A pipeline filter which pseudorandomly assigns values from the input into "train" or "test"
    groups.

    Examples:
        We can create complementary train and test filters using the function:
        >>> train_filter, test_filter = train_test_filter(train_p=0.6, seed=180)

        The train filter generates a sequence of 60% of the input list.
        >>> list(train_filter(range(20)))
        [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18, 19]

        When we run the test_filter, it fills in the gaps.
        >>> list(test_filter(range(20)))
        [1, 7, 8, 13, 14]

        We can continue to generate new values for as long as we like using the same filter and the
        continuation of the input range:
        >>> list(train_filter(range(20, 40)))
        [20, 22, 23, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39]

        ... and some more.
        >>> list(train_filter(range(40, 50)))
        [41, 42, 44, 45, 46, 49]

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

    """

    test_p = 1 - train_p

    _TrainTest = Enum("_TrainTest", ["train", "test"])

    def _train_test_stream():
        rng = np.random.default_rng(seed)
        while True:
            yield rng.choice([_TrainTest.train, _TrainTest.test], p=(train_p, test_p))

    def _factory(allow):
        _stream = _train_test_stream()

        def _generator(values):
            for v, train_test in zip(values, _stream):
                if train_test == allow:
                    yield v

        return _generator

    return _factory(_TrainTest.train), _factory(_TrainTest.test)
