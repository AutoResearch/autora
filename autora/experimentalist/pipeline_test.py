from itertools import product

import numpy as np

from .pipeline import PoolPipeline

##############################################################################
# Simple pool and filters of one variable
##############################################################################


def linear_pool_generator(stop=10):
    return range(stop)


def even_filter(values):
    return filter(lambda i: i % 2 == 0, values)


def odd_filter(values):
    return filter(lambda i: (i + 1) % 2 == 0, values)


def test_zeroth_pipeline():
    pipeline = PoolPipeline(linear_pool_generator)
    result = list(pipeline())
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_even_pipeline():
    pipeline = PoolPipeline(linear_pool_generator, even_filter)
    result = list(pipeline())
    assert result == [0, 2, 4, 6, 8]


def test_odd_pipeline():
    pipeline = PoolPipeline(linear_pool_generator, odd_filter)
    result = list(pipeline())
    assert result == [1, 3, 5, 7, 9]


def test_pipeline_run():
    pipeline = PoolPipeline(linear_pool_generator, odd_filter)
    result = list(pipeline.run())
    assert result == [1, 3, 5, 7, 9]


##############################################################################
# Pool and filters of two Weber variables
##############################################################################


def weber_pool(vmin=0, vmax=1, steps=5):
    s1 = s2 = np.linspace(vmin, vmax, steps)
    pool = product(s1, s2)
    return pool


def weber_filter(values):
    return filter(lambda s: s[0] >= s[1], values)


def test_weber_unfiltered_pipeline():
    pipeline = PoolPipeline(weber_pool)
    result = list(pipeline())
    assert result[0] == (0.0, 0.0)
    assert result[1] == (0.0, 0.25)
    assert result[-1] == (1.0, 1.0)


def test_weber_filtered_pipeline():
    pipeline = PoolPipeline(weber_pool, weber_filter)
    result = list(pipeline())
    assert result[0] == (0.0, 0.0)
    assert result[1] == (0.25, 0.0)
    assert result[-1] == (1.0, 1.0)
