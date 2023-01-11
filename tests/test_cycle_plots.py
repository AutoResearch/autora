import random

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from autora.cycle import (
    Cycle,
    cycle_default_score,
    cycle_specified_score,
    plot_cycle_score,
    plot_results_panel_2d,
    plot_results_panel_3d,
)
from autora.cycle.plot_utils import _check_replace_default_kw
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler.general_pool import grid_pool
from autora.experimentalist.sampler import random_sampler
from autora.variable import Variable, VariableCollection


@pytest.fixture
def ground_truth_1d():
    def ground_truth(xs):
        return xs + 1.0

    return ground_truth


@pytest.fixture
def cycle_lr(ground_truth_1d):
    random.seed(1)

    # Variable Metadata
    study_metadata = VariableCollection(
        independent_variables=[
            Variable(name="x1", allowed_values=np.linspace(0, 1, 100))
        ],
        dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    )

    # Theorist
    lm = LinearRegression()

    # Experimentalist
    example_experimentalist = Pipeline(
        [
            ("pool", grid_pool),
            ("sampler", random_sampler),
            ("transform", lambda x: [s[0] for s in x]),
        ],
        params={
            "pool": {"ivs": study_metadata.independent_variables},
            "sampler": {"n": 5},
        },
    )

    # Experiment Runner
    def get_example_synthetic_experiment_runner():
        rng = np.random.default_rng(seed=180)

        def runner(xs):
            return ground_truth_1d(xs) + rng.normal(0, 0.1, xs.shape)

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()

    # Initialize Cycle
    cycle = Cycle(
        metadata=study_metadata,
        theorist=lm,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
    )

    return cycle


@pytest.fixture
def ground_truth_2d():
    def ground_truth(X):
        return X[:, 0] + (0.5 * X[:, 1]) + 1.0

    return ground_truth


@pytest.fixture
def cycle_multi_lr(ground_truth_2d):
    random.seed(1)

    # def ground_truth(X):
    #     return X[:, 0] + (0.5 * X[:, 1]) + 1.0

    # Variable Metadata
    study_metadata = VariableCollection(
        independent_variables=[
            Variable(name="x1", allowed_values=np.linspace(0, 1, 10)),
            Variable(name="x2", allowed_values=np.linspace(0, 1, 10)),
        ],
        dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    )

    # Theorist
    lm = LinearRegression()

    # Experimentalist
    example_experimentalist = Pipeline(
        [
            ("pool", grid_pool),
            ("sampler", random_sampler),
            ("transform", lambda x: np.array(x)),
        ],
        params={
            "pool": {"ivs": study_metadata.independent_variables},
            "sampler": {"n": 10},
        },
    )

    # Experiment Runner
    def get_example_synthetic_experiment_runner():
        rng = np.random.default_rng(seed=180)

        def runner(xs):
            return ground_truth_2d(xs) + rng.normal(0, 0.25, xs.shape[0])

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()

    # Initialize Cycle
    cycle = Cycle(
        metadata=study_metadata,
        theorist=lm,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
    )

    return cycle


def test_check_replace_default_kw():
    default = {
        "subplot_kw": {"sharex": True, "sharey": True},
        "gridspec_kw": {"bottom": 0.16},
    }
    user = {
        "new_kw": True,
        "subplot_kw": {"sharey": False},
        "gridspec_kw": {"bottom": 0.2, "top": 0.9},
    }
    d_result = _check_replace_default_kw(default, user)

    assert d_result == {
        "subplot_kw": {"sharex": True, "sharey": False},
        "gridspec_kw": {"bottom": 0.2, "top": 0.9},
        "new_kw": True,
    }


def test_2d_plot(cycle_lr):
    """
    Tests the 2d plotting functionality of plot_results_panel.
    """
    cycle_lr.run(8)
    steps = 51
    fig = plot_results_panel_2d(
        cycle_lr, steps=steps, wrap=3, subplot_kw={"sharex": True, "sharey": True}
    )

    # Should have 9 axes, 8 with data and the last turned off
    assert len(fig.axes) == 9
    assert sum([s.axison for s in fig.axes]) == 8

    # Check number of data points on each figure
    # Blue dots should start at 0 and augment by 5.
    # Orange should always be 5-this is the condition sampling rate set by the Experimentalist.
    l_counts = []
    for axes in fig.axes[:-1]:
        blue_dots = (
            len(axes.collections[0].get_offsets().mask)
            - axes.collections[0].get_offsets().mask.any(axis=1).sum()
        )
        orange_dots = (
            len(axes.collections[1].get_offsets().mask)
            - axes.collections[1].get_offsets().mask.any(axis=1).sum()
        )
        l_counts.append([blue_dots, orange_dots])
    assert np.array_equal(
        np.array(l_counts),
        np.array(
            [[0, 5], [5, 5], [10, 5], [15, 5], [20, 5], [25, 5], [30, 5], [35, 5]]
        ),
    )

    # Test theory line is being plotted
    for axes in fig.axes[:-1]:
        assert len(axes.lines[0].get_xdata()) == steps
        assert len(axes.lines[0].get_ydata()) == steps


def test_3d_plot(cycle_multi_lr):
    cycle_multi_lr.run(6)
    steps = 20
    fig = plot_results_panel_3d(
        cycle_multi_lr,
        steps=steps,
        view=(20, 60),
        wrap=3,
        subplot_kw=dict(figsize=(11, 8)),
    )

    # Should have 6 axes
    assert len(fig.axes) == 6
    assert sum([s._axis3don for s in fig.axes]) == 6

    # Check number of data points on each figure
    # Blue dots should start at 0 and augment by 5.
    # Orange should always be 5-this is the condition sampling rate set by the Experimentalist.
    l_counts = []
    for axes in fig.axes[:-1]:
        blue_dots = (
            len(axes.collections[0].get_offsets().mask)
            - axes.collections[0].get_offsets().mask.any(axis=1).sum()
        )
        orange_dots = (
            len(axes.collections[1].get_offsets().mask)
            - axes.collections[1].get_offsets().mask.any(axis=1).sum()
        )
        l_counts.append([blue_dots, orange_dots])
    assert np.array_equal(
        np.array(l_counts),
        np.array([[0, 10], [10, 10], [20, 10], [30, 10], [40, 10]]),
    )


def test_score_functions(cycle_lr, ground_truth_1d):
    cycle_lr.run(10)
    X_test = cycle_lr.data.metadata.independent_variables[0].allowed_values.reshape(
        -1, 1
    )
    y_test = ground_truth_1d(X_test)

    scores_default = cycle_default_score(cycle_lr, X_test, y_test)
    scores_specified = cycle_specified_score(r2_score, cycle_lr, X_test, y_test)

    # Check scores the expected values
    score_values = [
        0.98950589,
        0.99352993,
        0.9858365,
        0.99909308,
        0.99811927,
        0.98663153,
        0.98748396,
        0.9848339,
        0.99359794,
        0.99691326,
    ]
    assert np.array_equal(np.around(score_values, 8), np.around(scores_default, 8))

    # Results should be equal between the two functions.
    # The default scorer of the LinearRegression estimator is r2_score.
    assert np.array_equal(scores_default, scores_specified)


def test_cycle_score_plot(cycle_lr, ground_truth_1d):
    cycle_lr.run(20)
    X_test = cycle_lr.data.metadata.independent_variables[0].allowed_values.reshape(
        -1, 1
    )
    y_test = ground_truth_1d(X_test)
    fig = plot_cycle_score(cycle_lr, X_test, y_test)

    # Should have 1 axis
    assert len(fig.axes) == 1

    # Test line is plotted correctly
    axis = fig.axes[0]
    assert len(axis.lines[0].get_xdata()) == 20
    y_values = np.array(
        [
            0.98950589,
            0.99352993,
            0.9858365,
            0.99909308,
            0.99811927,
            0.98663153,
            0.98748396,
            0.9848339,
            0.99359794,
            0.99691326,
            0.99547573,
            0.995913,
            0.99711678,
            0.99841886,
            0.99737463,
            0.9972299,
            0.99772379,
            0.99838647,
            0.99853528,
            0.99798914,
        ]
    )
    y_plotted = axis.lines[0].get_ydata()
    assert np.array_equal(np.around(y_plotted, 8), np.around(y_values, 8))


def test_cycle_score_plot_multi_lr(cycle_multi_lr, ground_truth_2d):
    cycle_multi_lr.run(12)
    X_test = np.array(
        list(grid_pool(cycle_multi_lr.data.metadata.independent_variables))
    )
    y_test = ground_truth_2d(X_test)
    fig = plot_cycle_score(cycle_multi_lr, X_test, y_test)

    # Test line is plotted correctly
    axis = fig.axes[0]
    assert len(axis.lines[0].get_xdata()) == 12
    y_values = np.array(
        [
            0.89368929,
            0.91897824,
            0.96375643,
            0.94514076,
            0.97807231,
            0.98778323,
            0.9931792,
            0.98768072,
            0.98952749,
            0.98867354,
            0.9872955,
            0.98999052,
        ]
    )
    y_plotted = axis.lines[0].get_ydata()
    assert np.array_equal(np.around(y_plotted, 8), np.around(y_values, 8))
