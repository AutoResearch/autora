"""
Example file which shows some simple curve fitting using DARTSRegressor and some other estimators.
"""

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aer.skl.darts import DARTSRegressor

matplotlib.use("module://backend_interagg")


# %% Define some helper functions


def get_plot_limits():
    plot_limits = plt.xlim(), plt.ylim()
    return plot_limits


def set_plot_limits(plot_limits):
    plt.xlim(plot_limits[0])
    plt.ylim(plot_limits[1])
    return


# Plot the results
def show_results_complete(data_: pd.DataFrame, estimator=None, show=True):

    ax = data_.plot.scatter(
        "EEG_Feature_A", "EEG_Feature_B", c="Diagnosis", cmap="viridis", zorder=10
    )

    if estimator is not None:

        # DecisionBoundaryDisplay.from_estimator modifies the plot range,
        # but we want the same plot range as set by the scatter diagram.
        # We get the plot limits here and reset them afterwards
        plot_limits = get_plot_limits()

        DecisionBoundaryDisplay.from_estimator(
            estimator,
            data_[["EEG_Feature_A", "EEG_Feature_B"]],
            plot_method="contourf",
            ax=ax,
            cmap="viridis_r",
            eps=0,
            alpha=0.5,
        )

        # Reset the plot limits
        set_plot_limits(plot_limits)

    if show:
        plt.show()


# %% Load the data

data = pd.read_csv("example/eeg/eeg_test_dataset.csv")
data["Diagnosis"] = pd.Categorical.from_codes(
    data["Diagnosis"], ordered=False, categories=["healthy", "diagnosed"]
)

show_results = partial(show_results_complete, data_=data)
show_results()

# %%

X = data[["EEG_Feature_A", "EEG_Feature_B"]]
y = data["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %% Fit first using a logistic regression

logistic_estimator = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        # There are fewer "healthy" than "diagnosed" data, so we balance the classes
        class_weight="balanced"
    ),
)

logistic_estimator.fit(X_train, y_train)

show_results(estimator=logistic_estimator)

# %% Fit using DARTS
darts_estimator = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        DARTSRegressor(
            batch_size=20,
        ),
    ),
    param_grid=dict(darts__num_graph_nodes=range(1, 5)),
)
darts_estimator.fit(X_train, y_train)

show_results(estimators={"DARTSRegressor": darts_estimator})
print(darts_estimator.best_params_)

# %%
mlp_estimator = make_pipeline(StandardScaler(), MLPClassifier())
mlp_estimator.fit(X_train, y_train)

show_results(estimator=mlp_estimator)
