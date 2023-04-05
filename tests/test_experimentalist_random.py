from functools import partial

import numpy as np
import pytest

from autora.experimentalist.pipeline import make_pipeline
from autora.experimentalist.pooler.grid import grid_pool
from autora.experimentalist.sampler.random_ import random_sampler
from autora.variable import DV, IV, ValueType, VariableCollection


def weber_filter(values):
    return filter(lambda s: s[0] <= s[1], values)


def test_random_experimentalist(metadata):
    """
    Tests the implementation of the experimentalist pipeline with an exhaustive pool of discrete
    values, Weber filter, random selector. Tests two different implementations of the pool function
    as a callable and passing in as interator/generator.

    """

    n_trials = 25  # Number of trails for sampler to select

    # ---Implementation 1 - Pool using Callable via partial function----
    # Set up pipeline functions with partial
    pooler_callable = partial(grid_pool, ivs=metadata.independent_variables)
    sampler = partial(random_sampler, n=n_trials)
    pipeline_random_samp = make_pipeline(
        [pooler_callable, weber_filter, sampler],
    )

    results = pipeline_random_samp.run()

    # ***Checks***
    # Gridsearch pool is working as expected
    _, pool = pipeline_random_samp.steps[0]
    pool_len = len(list(pool()))
    pool_len_expected = np.prod(
        [len(s.allowed_values) for s in metadata.independent_variables]
    )
    assert pool_len == pool_len_expected

    # Is sampling the number of trials we expect
    assert len(results) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] <= s[1] for s in results])

    # Is sampling randomly. Runs 10 times and checks if consecutive runs are equal.
    # Assert will fail if all 9 pairs return equal.
    l_results = [pipeline_random_samp.run() for s in range(10)]
    assert not np.all(
        [
            np.array_equal(l_results[i], l_results[i + 1])
            for i, s in enumerate(l_results)
            if i < len(l_results) - 1
        ]
    )


def test_random_experimentalist_generator(metadata):
    n_trials = 25  # Number of trails for sampler to select

    pooler_generator = grid_pool(metadata.independent_variables)
    sampler = partial(random_sampler, n=n_trials)
    pipeline_random_samp_poolgen = make_pipeline(
        [pooler_generator, weber_filter, sampler]
    )

    results_poolgen = list(pipeline_random_samp_poolgen.run())

    # Is sampling the number of trials we expect
    assert len(results_poolgen) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] <= s[1] for s in results_poolgen])

    # This will fail
    # The Generator is exhausted after the first run and the pool is not regenerated when pipeline
    # is run again. The pool should be set up as a callable if the pipeline is to be rerun.
    results_poolgen2 = pipeline_random_samp_poolgen.run()
    assert len(results_poolgen2) == 0


@pytest.fixture
def metadata():
    # Specify independent variables
    iv1 = IV(
        name="S1",
        allowed_values=np.linspace(0, 5, 5),
        units="intensity",
        variable_label="Stimulus 1 Intensity",
    )

    iv2 = IV(
        name="S2",
        allowed_values=np.linspace(0, 5, 5),
        units="intensity",
        variable_label="Stimulus 2 Intensity",
    )

    iv3 = IV(
        name="S3",
        allowed_values=[0, 1],
        units="binary",
        variable_label="Stimulus 3 Binary",
    )

    # Specify dependent variable with type
    # The experimentalist pipeline doesn't actually use DVs, they are just specified here for
    # example.
    dv1 = DV(
        name="difference_detected",
        value_range=(0, 1),
        units="probability",
        variable_label="P(difference detected)",
        type=ValueType.SIGMOID,
    )

    dv2 = DV(
        name="difference_detected_sample",
        value_range=(0, 1),
        units="response",
        variable_label="difference detected",
        type=ValueType.PROBABILITY_SAMPLE,
    )
    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv2, iv3],
        dependent_variables=[dv1, dv2],
    )

    return metadata
