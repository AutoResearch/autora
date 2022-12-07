import random

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from autora.cycle import Cycle
from autora.cycle.plot import plot_results_panel
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pool import grid_pool
from autora.experimentalist.sampler import random_sampler
from autora.variable import Variable, VariableCollection


@pytest.fixture
def dummy_cycle():
    random.seed(1)

    def ground_truth(xs):
        return xs + 1.0

    study_metadata = VariableCollection(
        independent_variables=[
            Variable(name="x1", allowed_values=np.linspace(0, 1, 100))
        ],
        dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    )

    grid_pool(study_metadata.independent_variables)
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

    def get_example_synthetic_experiment_runner():
        rng = np.random.default_rng(seed=180)

        def runner(xs):
            return ground_truth(xs) + rng.normal(0, 0.1, xs.shape)

        return runner

    example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()
    lm = LinearRegression()

    cycle = Cycle(
        metadata=study_metadata,
        theorist=lm,
        experimentalist=example_experimentalist,
        experiment_runner=example_synthetic_experiment_runner,
    )

    return cycle


def test_2d_plot(dummy_cycle):
    dummy_cycle.run(8)
    steps = 51
    fig = plot_results_panel(dummy_cycle, steps=steps, sharex=True, sharey=True, wrap=3)

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
