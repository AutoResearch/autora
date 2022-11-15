import collections
import pathlib
import random
from functools import partial
from itertools import product
from typing import List

import numpy as np

from autora.experimentalist.pipeline import Pipeline
from autora.variable import DV, IV, ValueType, VariableCollection

# import pandas as pd
# from alipy.query_strategy.query_labels import QueryInstanceUncertainty


class Experimentalist:
    def __init__(self, pool_method, sampling_method, **args):
        self._pool_method = pool_method
        self._sampling_method = sampling_method


def random_product(*args, repeat=1):
    """Random selection from itertools.product(*args, **kwds)"""
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(map(random.choice, pools))


def gridsearch_pool(ivs: List[IV]):
    """Returns Cartesian product of sets"""
    # Get allowed values for each IV
    l_iv_values = [iv.allowed_values for iv in ivs]

    # Return Cartesian product of all IV values
    return product(*l_iv_values)


def random_sampler(values, n):
    if isinstance(values, collections.Iterable):
        values = list(values)
    random.shuffle(values)
    samples = values[0:n]

    return samples


def weber_filter(values):
    return filter(lambda s: s[0] >= s[1], values)


def uncertainty_sampler(
    model,
):
    model
    # Theory
    # n samples to select


def test_experimentalist():
    # %% Load the data
    # datafile_path = pathlib.Path(__file__).parent.parent.joinpath(
    #     "example/sklearn/darts/weber_data.csv"
    # )
    # data = pd.read_csv(datafile_path)

    # specify independent variables
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

    # specify dependent variable with type
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

    metadata = VariableCollection(
        independent_variables=[iv1, iv2, iv3],
        dependent_variables=[dv1, dv2],
    )

    n_trials = 25

    # Set up pipeline functions with partial
    pooler_callable = partial(gridsearch_pool, ivs=metadata.independent_variables)
    sampler = partial(random_sampler, n=n_trials)

    pipeline_random_samp = Pipeline(
        pooler_callable,
        weber_filter,
        sampler,
    )

    results = pipeline_random_samp.run()

    # ***Checks***
    # Gridsearch pool is working as expected
    pool_len = len(list(pipeline_random_samp.pool()))
    pool_len_expected = np.prod(
        [len(s.allowed_values) for s in metadata.independent_variables]
    )
    assert pool_len == pool_len_expected

    # Is sampling the number of trials we expect
    assert len(results) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] >= s[1] for s in results])

    # Is sampling randomly
    results2 = pipeline_random_samp.run()
    assert results != results2

    # ---Pool using Generator----
    pooler_generator = gridsearch_pool(metadata.independent_variables)
    pipeline_random_samp_poolgen = Pipeline(
        pooler_generator,
        weber_filter,
        sampler,
    )

    results_poolgen = pipeline_random_samp_poolgen.run()

    # Is sampling the number of trials we expect
    assert len(results_poolgen) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] >= s[1] for s in results_poolgen])

    # This will fail. Generator is exhausted and pool is not regenerated when pipeline is run again.
    results_poolgen2 = pipeline_random_samp_poolgen.run()
    assert len(results_poolgen2) > 0


if __name__ == "__main__":
    test_experimentalist()
