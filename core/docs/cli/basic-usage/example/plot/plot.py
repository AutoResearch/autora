#!/usr/bin/env python
import pathlib

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

from autora.state import StandardState
from autora.workflow.__main__ import load_state

from .lib import ground_truth, noise_std


def plot_results(state: StandardState):
    x = np.linspace(-10, 10, 100).reshape((-1, 1))
    plt.plot(x, ground_truth(x), label="ground_truth", c="orange")
    plt.fill_between(
        x.flatten(),
        ground_truth(x).flatten() + noise_std,
        ground_truth(x).flatten() - noise_std,
        alpha=0.3,
        color="orange",
    )

    assert isinstance(state.experiment_data, pd.DataFrame)
    xi, yi = state.experiment_data["x"], state.experiment_data["y"]
    plt.scatter(xi, yi, label="observations")

    assert isinstance(state.models[-1], GridSearchCV)
    plt.plot(x, state.models[-1].predict(x), label="model")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()
    plt.show()


def main(filename: pathlib.Path):
    state = load_state(filename)
    assert isinstance(state, StandardState)
    plot_results(state)


if __name__ == "__main__":
    typer.run(main)
