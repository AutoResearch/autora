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
from autora.experimentalist.pooler.grid import grid_pool
from autora.experimentalist.sampler.random_ import random_sampler
from autora.variable import Variable, VariableCollection


@pytest.fixture
def ground_truth_1x():
    def ground_truth(xs):
        return xs + 1.0

    return ground_truth


@pytest.fixture
def cycle_lr(ground_truth_1x):
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
            return ground_truth_1x(xs) + rng.normal(0, 0.1, xs.shape)

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()

    # Initialize Cycle
    cycle = Cycle(
        metadata=study_metadata,
        theorist=lm,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
    )

    # Run 10 iterations
    cycle.run(10)

    return cycle


@pytest.fixture
def ground_truth_2x():
    def ground_truth(X):
        return X[:, 0] + (0.5 * X[:, 1]) + 1.0

    return ground_truth


@pytest.fixture
def cycle_multi_lr(ground_truth_2x):
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
            return ground_truth_2x(xs) + rng.normal(0, 0.25, xs.shape[0])

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()

    # Initialize Cycle
    cycle = Cycle(
        metadata=study_metadata,
        theorist=lm,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
    )

    # Run 6 iterations
    cycle.run(6)

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
    Tests plotting functionality of plot_results_panel_2d.
    """
    steps = 51
    fig = plot_results_panel_2d(
        cycle_lr, steps=steps, wrap=3, subplot_kw={"sharex": True, "sharey": True}
    )

    # Should have 12 axes, 10 with data and the last 2 turned off
    assert len(fig.axes) == 12
    assert sum([s.axison for s in fig.axes]) == 10

    # Check number of data points on each figure
    # Blue dots should start at 0 and augment by 5.
    # Orange should always be 5-this is the condition sampling rate set by the Experimentalist.
    l_counts = []
    for axes in fig.axes[:-2]:
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
            [
                [0, 5],
                [5, 5],
                [10, 5],
                [15, 5],
                [20, 5],
                [25, 5],
                [30, 5],
                [35, 5],
                [40, 5],
                [45, 5],
            ]
        ),
    )

    # Test theory line is being plotted
    for axes in fig.axes[:-2]:
        assert len(axes.lines[0].get_xdata()) == steps
        assert len(axes.lines[0].get_ydata()) == steps


def test_3d_plot(cycle_multi_lr):
    """
    Tests plotting functionality of plot_results_panel_3d.
    """
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


def test_score_functions(cycle_lr, ground_truth_1x):
    """
    Tests the scoring functions cycle_default_score and cycle_specified_score.
    """
    X_test = cycle_lr.data.metadata.independent_variables[0].allowed_values.reshape(
        -1, 1
    )
    y_test = ground_truth_1x(X_test)

    scores_default = cycle_default_score(cycle_lr, X_test, y_test)
    scores_specified = cycle_specified_score(r2_score, cycle_lr, X_test, y_test)

    # Check scores are the expected values
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


def test_cycle_score_plot(cycle_lr, ground_truth_1x):
    """
    Tests plotting functionality of test_cycle_score_plot with a 2D linear regression.
    """
    X_test = cycle_lr.data.metadata.independent_variables[0].allowed_values.reshape(
        -1, 1
    )
    y_test = ground_truth_1x(X_test)
    fig = plot_cycle_score(cycle_lr, X_test, y_test)

    # Should have 1 axis
    assert len(fig.axes) == 1

    # Test line is plotted correctly
    axis = fig.axes[0]
    assert len(axis.lines[0].get_xdata()) == 10
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
        ]
    )
    y_plotted = axis.lines[0].get_ydata()
    assert np.array_equal(np.around(y_plotted, 8), np.around(y_values, 8))


def test_cycle_score_plot_multi_lr(cycle_multi_lr, ground_truth_2x):
    """
    Tests plotting functionality of test_cycle_score_plot with multiple linear regression cycle.
    """
    cycle_multi_lr.run(6)  # Run additional 6 times, total of 12 cycles
    X_test = np.array(
        list(grid_pool(cycle_multi_lr.data.metadata.independent_variables))
    )
    y_test = ground_truth_2x(X_test)
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


def test_2d_plot_indexing(cycle_lr):
    """
    Test indexing of 2d plotter.
    """
    steps = 51
    fig = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=2,
        query=[0, 3, 7],
        subplot_kw={"sharex": True, "sharey": True},
    )

    # Should have 4 axes, 3 with data and the last turned off
    assert len(fig.axes) == 4
    assert sum([s.axison for s in fig.axes]) == 3


def test_2d_plot_negative_indexing(cycle_lr):
    """
    Test indexing of 2d plotter.
    """
    steps = 51
    fig = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=2,
        query=[-2, -1],
        subplot_kw={"sharex": True, "sharey": True},
    )

    # Should have 2 axes
    assert len(fig.axes) == 2
    assert sum([s.axison for s in fig.axes]) == 2

    # Should be plotting cycles 8 and 9
    assert fig.axes[0].get_children()[3].get_text() == "Cycle 8"
    assert fig.axes[1].get_children()[3].get_text() == "Cycle 9"


def test_2d_plot_slicing(cycle_lr):
    """
    Test slicing of 2d plotter using built-in slice() function.
    """
    steps = 51

    # Using Slice function
    # Cycles 0, 2, 4, 6, 8
    fig = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=3,
        query=slice(0, 9, 2),
        subplot_kw={"sharex": True, "sharey": True},
    )
    # Should have 6 axes, 5 with data and the last turned off
    assert len(fig.axes) == 6
    assert sum([s.axison for s in fig.axes]) == 5

    # Last 4 plots
    fig2 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=3,
        query=slice(-4, None, None),
        subplot_kw={"sharex": True, "sharey": True},
    )
    # Should have 6 axes, 4 with data
    assert len(fig2.axes) == 6
    assert sum([s.axison for s in fig2.axes]) == 4


def test_2d_plot_slicing_np(cycle_lr):
    """
    Test slicing of 2d plotter using np.s_ Index Expression
    """
    steps = 51

    # Cycles 0, 2, 4, 6, 8
    fig1 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=3,
        query=np.s_[0:9:2],
        subplot_kw={"sharex": True, "sharey": True},
    )
    # Should have 6 axes, 5 with data and the last turned off
    assert len(fig1.axes) == 6
    assert sum([s.axison for s in fig1.axes]) == 5

    fig2 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        wrap=3,
        query=np.s_[-4:],
        subplot_kw={"sharex": True, "sharey": True},
    )
    # Should have 6 axes, 4 with data
    assert len(fig2.axes) == 6
    assert sum([s.axison for s in fig2.axes]) == 4


def test_2d_plot_plot_single(cycle_lr):
    """
    Test query of 2d plotter for a single cycle.
    """
    steps = 51

    # Using index
    fig1 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        query=[9],
        subplot_kw={"sharex": True, "sharey": True},
    )
    assert len(fig1.axes) == 1
    assert sum([s.axison for s in fig1.axes]) == 1

    # Using slice()
    fig2 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        query=slice(-1, None, None),
        subplot_kw={"sharex": True, "sharey": True},
    )
    assert len(fig2.axes) == 1
    assert sum([s.axison for s in fig2.axes]) == 1

    # Using np.s_ Index expression
    fig3 = plot_results_panel_2d(
        cycle_lr,
        steps=steps,
        query=np.s_[-1:],
        subplot_kw={"sharex": True, "sharey": True},
    )
    assert len(fig3.axes) == 1
    assert sum([s.axison for s in fig3.axes]) == 1
