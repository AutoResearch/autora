import pathlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from autora.experimentalist.filter import weber_filter
from autora.experimentalist.pipeline import make_pipeline
from autora.experimentalist.pooler.general_pool import grid_pool
from autora.experimentalist.sampler import uncertainty_sampler
from autora.variable import DV, IV, ValueType, VariableCollection


def test_experimentalist_uncertainty():
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
    pooler_callable = partial(grid_pool, ivs=metadata.independent_variables)
    sampler = partial(
        uncertainty_sampler, model=logireg_model, n=n_trials, measure="least_confident"
    )

    # Initialize pipeline
    pipeline = make_pipeline([pooler_callable, weber_filter, sampler])
    # Run the pipeline
    results = pipeline.run()

    # ***Checks***
    # Is sampling the number of trials we expect
    assert len(results) == n_trials

    # Filter is selecting where IV1 >= IV2
    assert all([s[0] <= s[1] for s in results])

    # Uncertainty sampling is behaving as expected by comparing results with manual calculation
    pipeline_pool_filter = make_pipeline([pooler_callable, weber_filter])
    pool = np.array(list(pipeline_pool_filter.run()))  # Create filtered pool
    a_prob = logireg_model.predict_proba(pool)  # Get predicted probabilities
    # Calculate and sort max probability from each condition
    s_max_prob = pd.Series([np.max(s) for s in a_prob]).sort_values(ascending=True)
    select_idx = s_max_prob.index[
        0:n_trials
    ].to_list()  # Get index of lowest probabilities
    results_manual = pool[select_idx]  # Index conditions from pool
    # Check results from the function match manaual method
    assert np.array_equal(results, results_manual)
