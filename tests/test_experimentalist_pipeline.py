from functools import partial
from itertools import product
from math import sqrt

import numpy as np

from autora.experimentalist.pipeline import (
    Pipeline,
    _parse_params_to_nested_dict,
    make_pipeline,
)

##############################################################################
# Building blocks
##############################################################################


def linear_pool_generator(stop=10):
    return range(stop)


def even_filter(values):
    return filter(lambda i: i % 2 == 0, values)


def odd_filter(values):
    return filter(lambda i: (i + 1) % 2 == 0, values)


def divisor_filter(values, divisor):
    return filter(lambda i: i % divisor == 0, values)


def is_sqrt_filter(values):
    return filter(lambda i: sqrt(i) % 1 == 0.0, values)


##############################################################################
# Simple pipelines of one variable
##############################################################################


def test_zeroth_pipline_zeroth_input():
    pipeline = Pipeline()
    result_0 = list(pipeline())
    assert result_0 == []


def test_zeroth_pipline_basic_input():
    pipeline = Pipeline([])
    result_0 = list(pipeline([0, 1, 2, 3]))
    assert result_0 == [0, 1, 2, 3]


def test_zeroth_make_pipeline():
    pipeline = make_pipeline()
    result = list(pipeline([0, 1, 2, 3]))
    assert result == [0, 1, 2, 3]


def test_single_element_pipeline():
    pipeline = Pipeline([("even_filter", even_filter)])
    result = list(pipeline(range(10)))
    assert result == [0, 2, 4, 6, 8]


def test_single_element_make_pipeline():
    pipeline = make_pipeline([even_filter])
    result = list(pipeline(range(10)))
    assert result == [0, 2, 4, 6, 8]


def test_multiple_element_pipeline():
    pipeline = Pipeline(
        [
            ("even_filter", even_filter),
            ("divisor_filter", partial(divisor_filter, divisor=3)),
        ]
    )
    result = list(pipeline(range(13)))
    assert result == [0, 6, 12]


def test_multiple_element_make_pipeline():
    pipeline = make_pipeline([even_filter, partial(divisor_filter, divisor=3)])
    result = list(pipeline(range(13)))
    assert result == [0, 6, 12]


def test_two_element_make_pipeline_with_params():
    pipeline = make_pipeline(
        [even_filter, divisor_filter], params={"divisor_filter": {"divisor": 5}}
    )
    result = list(pipeline(range(21)))
    assert result == [0, 10, 20]


def test_three_element_make_pipeline():
    pipeline = make_pipeline(
        [divisor_filter, divisor_filter, divisor_filter],
        params={
            "divisor_filter_0": {"divisor": 5},
            "divisor_filter_1": {"divisor": 7},
            "divisor_filter_2": {"divisor": 11},
        },
    )
    result = list(pipeline(range(500)))
    assert result == [0, 385]


def test_nested_pipeline():
    inner_pipeline = Pipeline([("pool", lambda: range(32))])
    outer_pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("filter_by_divisor", partial(divisor_filter, divisor=8)),
        ],
    )
    result = list(outer_pipeline())
    assert result == [0, 8, 16, 24]


def test_nested_pipeline_nested_parameters():
    inner_pipeline = Pipeline([("pool", lambda maximum: range(maximum))])
    outer_pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("filter_by_divisor", divisor_filter),
        ],
        params={
            "inner_pipeline": {"pool": {"maximum": 32}},
            "filter_by_divisor": {"divisor": 8},
        },
    )
    result = list(outer_pipeline())
    assert result == [0, 8, 16, 24]


def test_nested_pipeline_flat_parameters():
    inner_pipeline = Pipeline([("pool", lambda maximum: range(maximum))])
    outer_pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("filter_by_divisor", divisor_filter),
        ],
        params={"inner_pipeline__pool__maximum": 32, "filter_by_divisor__divisor": 8},
    )
    result = list(outer_pipeline())
    assert result == [0, 8, 16, 24]


def test_nested_pipeline_nested_parameters_in_kwargs():
    inner_pipeline = Pipeline([("pool", lambda maximum: range(maximum))])
    outer_pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("filter_by_divisor", divisor_filter),
        ],
    )
    result = list(
        outer_pipeline(
            **{
                "inner_pipeline": {"pool": {"maximum": 32}},
                "filter_by_divisor": {"divisor": 8},
            }
        )
    )
    assert result == [0, 8, 16, 24]


def test_nested_pipeline_flat_parameters_in_kwargs():
    inner_pipeline = Pipeline([("pool", lambda maximum: range(maximum))])
    outer_pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("filter_by_divisor", divisor_filter),
        ],
    )
    result = list(
        outer_pipeline(
            **{"inner_pipeline__pool__maximum": 32, "filter_by_divisor__divisor": 8}
        )
    )
    assert result == [0, 8, 16, 24]


##############################################################################
# Simple pool and filters of one variable
##############################################################################


def test_zeroth_poolpipeline():
    pipeline = make_pipeline([linear_pool_generator])
    result = list(pipeline())
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_even_poolpipeline():
    pipeline = make_pipeline([linear_pool_generator, even_filter])
    result = list(pipeline())
    assert result == [0, 2, 4, 6, 8]


def test_odd_poolpipeline():
    pipeline = make_pipeline([linear_pool_generator, odd_filter])
    result = list(pipeline())
    assert result == [1, 3, 5, 7, 9]


def test_poolpipeline_run():
    pipeline = make_pipeline([linear_pool_generator, odd_filter])
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


def test_weber_unfiltered_poolpipeline():
    pipeline = make_pipeline([weber_pool])
    result = list(pipeline())
    assert result[0] == (0.0, 0.0)
    assert result[1] == (0.0, 0.25)
    assert result[-1] == (1.0, 1.0)


def test_weber_filtered_poolpipeline():
    pipeline = make_pipeline([weber_pool, weber_filter])
    result = list(pipeline())
    assert result[0] == (0.0, 0.0)
    assert result[1] == (0.25, 0.0)
    assert result[-1] == (1.0, 1.0)


##############################################################################
# Helper Functions
##############################################################################
def test_params_parser_zero_level():
    params = {
        "model": "%%newest_theory%%",
        "n": 10,
        "measure": "least_confident",
    }
    result = _parse_params_to_nested_dict(params, divider="__")
    assert result == params


def test_params_parser_one_level():
    params = {
        "pool__ivs": "%%independent_variables%%",
        "uncertainty_sampler__model": "%%newest_theory%%",
        "uncertainty_sampler__n": 10,
        "uncertainty_sampler__measure": "least_confident",
    }

    result = _parse_params_to_nested_dict(params, divider="__")
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

    result = _parse_params_to_nested_dict(params, divider="__")
    assert result == {"filter_pipeline": {"step1": {"n_samples": 100}}}


def test_params_parser_recurse_one_n_levels():
    params = {
        "a__b__c__d__e__f": 100,
        "a__b__c__d__e__g": 200,
        "a__b__h": 300,
    }
    result = _parse_params_to_nested_dict(params, divider="__")
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

    result = _parse_params_to_nested_dict(params, divider="__")
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


##############################################################################
# Parallel Pipelines
##############################################################################


def test_parallelpipeline_run():
    pl = make_pipeline([range(3), range(10, 13)], kind="union")
    assert list(pl.run()) == [0, 1, 2, 10, 11, 12]


def test_parallelpipeline_many_steps():
    pl = make_pipeline([range(0, 5) for _ in range(1000)], kind="union")
    results = list(pl.run())
    assert len(results) == 5000
    assert results[0:10] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
