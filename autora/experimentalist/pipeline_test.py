from itertools import product

import numpy as np

from .pipeline import PoolPipeline, _parse_params_to_nested_dict

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


def test_params_parser_zero_level():
    params = {
        "model": "%%newest_theory%%",
        "n": 10,
        "measure": "least_confident",
    }
    result = _parse_params_to_nested_dict(params)
    assert result == params


def test_params_parser_one_level():
    params = {
        "pool__ivs": "%%independent_variables%%",
        "uncertainty_sampler__model": "%%newest_theory%%",
        "uncertainty_sampler__n": 10,
        "uncertainty_sampler__measure": "least_confident",
    }

    result = _parse_params_to_nested_dict(params)
    assert result == {
        "pool": {
            "ivs": "%%independent_variables%%",
        },
        "uncertainty_sampler": {
            "model": "%%newest_theory%%",
            "n": 10,
            "measure": "least_confident",
        },
    }


def test_params_parser_recurse_one():

    params = {
        "filter_pipeline__step1__n_samples": 100,
    }

    result = _parse_params_to_nested_dict(params)
    assert result == {"filter_pipeline": {"step1": {"n_samples": 100}}}


def test_params_parser_recurse_one_n_levels():
    params = {
        "a__b__c__d__e__f": 100,
        "a__b__c__d__e__g": 200,
        "a__b__h": 300,
    }
    result = _parse_params_to_nested_dict(params)
    assert result == {"a": {"b": {"c": {"d": {"e": {"f": 100, "g": 200}}}, "h": 300}}}


def test_params_parser_recurse_one_n_levels_alternative_divider():
    params = {
        "a:b:c:d:e:f": 100,
        "a:b:c:d:e:g": 200,
        "a:b:h": 300,
    }
    result = _parse_params_to_nested_dict(params, divider=":")
    assert result == {"a": {"b": {"c": {"d": {"e": {"f": 100, "g": 200}}}, "h": 300}}}


def test_params_parser_recurse():

    params = {
        "pool__ivs": "%%independent_variables%%",
        "filter_pipeline__step1__n_samples": 100,
        "filter_pipeline__step2__n_samples": 10,
        "uncertainty_sampler__model": "%%newest_theory%%",
        "uncertainty_sampler__n": 10,
        "uncertainty_sampler__measure": "least_confident",
    }

    result = _parse_params_to_nested_dict(params)
    assert result == {
        "pool": {
            "ivs": "%%independent_variables%%",
        },
        "filter_pipeline": {"step1": {"n_samples": 100}, "step2": {"n_samples": 10}},
        "uncertainty_sampler": {
            "model": "%%newest_theory%%",
            "n": 10,
            "measure": "least_confident",
        },
    }
