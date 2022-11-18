import pathlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from autora.experimentalist.filter import weber_filter
from autora.experimentalist.pipeline import PoolPipeline
from autora.experimentalist.pool import gridsearch_pool
from autora.experimentalist.sampler import random_sampler, uncertainty_sampler
from autora.variable import DV, IV, ValueType, VariableCollection


def test_random_experimentalist():
    """
    Tests the implementation of the experimentalist pipeline with an exhaustive pool of discrete
    values, Weber filter, random selector. Tests two different implementations of the pool function
    as a callable and passing in as interator/generator.

    """

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

    n_trials = 25  # Number of trails for sampler to select

    # ---Implementation 1 - Pool using Callable via partial function----
    # Set up pipeline functions with partial
    pooler_callable = partial(gridsearch_pool, ivs=metadata.independent_variables)
    sampler = partial(random_sampler, n=n_trials)
    pipeline_random_samp = PoolPipeline(
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

    # ---Implementation 2 - Pool using Generator----
    pooler_generator = gridsearch_pool(metadata.independent_variables)
    pipeline_random_samp_poolgen = PoolPipeline(
        pooler_generator,
        weber_filter,
        sampler,
    )

    results_poolgen = pipeline_random_samp_poolgen.run()

    # Is sampling the number of trials we expect
    assert len(results_poolgen) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] <= s[1] for s in results_poolgen])

    # This will fail
    # The Generator is exhausted after the first run and the pool is not regenerated when pipeline
    # is run again. The pool should be set up as a callable if the pipeline is to be rerun.
    results_poolgen2 = pipeline_random_samp_poolgen.run()
    assert len(results_poolgen2) > 0


def test_uncertainty_experimentalist():
    """
    Tests the implementation of the experimentalist pipeline with an exhaustive pool of discrete
    values, Weber filter, uncertainty sampler. A logistic regression model is trained using
    synthetic Weber experiment data for use in Uncertainty sampling.

    """
    # Load the data
    datafile_path = pathlib.Path(__file__).parent.parent.joinpath(
        "example/sklearn/darts/weber_data.csv"
    )
    data = pd.read_csv(datafile_path)
    X = data[["S1", "S2"]]
    y = data["difference_detected"]
    y_classified = np.where(y >= 0.5, 1, 0)

    # Train logistic regression model
    logireg_model = LogisticRegression()
    logireg_model.fit(X, y_classified)

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

    # The experimentalist pipeline doesn't actually use DVs, they are just specified here for
    # example.
    dv1 = DV(
        name="difference_detected",
        value_range=(0, 1),
        units="probability",
        variable_label="P(difference detected)",
        type=ValueType.PROBABILITY,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    n_trials = 10  # Number of trails for sampler to select

    # Set up pipeline functions with partial
    pooler_callable = partial(gridsearch_pool, ivs=metadata.independent_variables)
    sampler = partial(uncertainty_sampler, model=logireg_model, n=n_trials)

    # Initialize pipeline
    pipeline = PoolPipeline(
        pooler_callable,
        weber_filter,
        sampler,
    )
    # Run the pipeline
    results = pipeline.run()

    # ***Checks***
    # Is sampling the number of trials we expect
    assert len(results) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] <= s[1] for s in results])

    # Uncertainty sampling is behaving as expected by comparing results with manual calculation
    pipeline_pool_filter = PoolPipeline(pooler_callable, weber_filter)
    pool = np.array(list(pipeline_pool_filter.run()))  # Create filtered pool
    a_prob = logireg_model.predict_proba(pool)  # Get predicted probabilities
    # Calculate and sort max probability from each condition
    s_max_prob = pd.Series([np.max(s) for s in a_prob]).sort_values(ascending=True)
    select_idx = s_max_prob.index[
        0:n_trials
    ].to_list()  # Get index of lowest probabilities
    results_manual = np.flip(pool[select_idx], axis=0)  # Index conditions from pool
    # Check results from the function match manaual method
    assert np.array_equal(results, results_manual)


if __name__ == "__main__":
    test_random_experimentalist()
    test_uncertainty_experimentalist()
