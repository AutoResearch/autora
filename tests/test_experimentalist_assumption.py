from functools import partial

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from autora.experimentalist.pipeline import make_pipeline
from autora.experimentalist.pooler import grid_pool
from autora.experimentalist.sampler import assumption_sampler
from autora.variable import Variable, VariableCollection


def test_experimentalist_assumption():
    """
    Tests the implementation of the experimentalist pipeline with an exhaustive pool of discrete
    values, no filter, assumption sampler. A ridge regression model is trained using
    synthetic data_closed_loop generated from a ground_truth function for use.

    """
    # Make the data_closed_loop
    def ground_truth(xs):
        return (xs**2.0) + xs + 1.0

    # X = np.random.randint(low=0, high=10, size=100).reshape(-1, 1)
    X = np.array(range(11)).reshape(-1, 1)
    y = np.array([x for x in ground_truth(X)])

    # Train ridge regression model
    model = RidgeCV(scoring="neg_mean_squared_error")
    model.fit(X, y)

    metadata = VariableCollection(
        independent_variables=[Variable(name="x1", allowed_values=range(11))],
        dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    )

    n_trials = 10  # Number of trails for sampler to select

    # Set up pipeline functions with partial
    pooler_callable = partial(grid_pool, ivs=metadata.independent_variables)
    sampler = partial(assumption_sampler, y=y, model=model, n=n_trials)
    # currently theorist objects are not required to have its loss function as an attribute

    # Initialize pipeline
    pipeline = make_pipeline([pooler_callable, sampler])
    # Run the pipeline
    results = pipeline.run()

    # ***Checks***
    # Is sampling the number of trials we expect
    assert len(results) == n_trials

    # Assumption sampling is behaving as expected by comparing results with manual calculation
    pipeline_pool_filter = make_pipeline([pooler_callable])
    pool = np.array(list(pipeline_pool_filter.run()))  # Create filtered pool
    y_pred = model.predict(X)  # get model predictions
    # model_loss_function = get_scorer(scoring=model.get_params()['scoring'])
    # assert model_loss_function == mean_squared_error
    # sklearn does not quite seem to have functionality to retrieve loss function used by model
    # you can get the scorer which corresponds to the loss function
    # but there does not seem to be any built-in dictionary to get the corresponding function
    model_loss_function = mean_squared_error
    model_loss = model_loss_function(
        y_true=y.reshape(1, -1), y_pred=y_pred.reshape(1, -1), multioutput="raw_values"
    )
    # calculate model loss on an alternative loss function
    alternative_model_loss_function = mean_absolute_error
    alternative_model_loss = alternative_model_loss_function(
        y_true=y.reshape(1, -1), y_pred=y_pred.reshape(1, -1), multioutput="raw_values"
    )
    loss_delta = alternative_model_loss - model_loss
    select_idx = np.flip(loss_delta.argsort()[:n_trials])
    results_manual = pool[select_idx]  # Index conditions from pool
    assert np.array_equal(results, results_manual)
