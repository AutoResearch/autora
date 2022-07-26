"""
Example file which shows some simple curve fitting using DARTS and some other estimators.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from aer.skl.darts import DARTS, ValueType

# %% Define some helper functions


def show_results_complete(
    data_: pd.DataFrame,
    estimator=None,
    show_results=True,
    projection="2d",
    label=None,
):
    """
    Function to plot input data (x_, y_) and the predictions of an estimator for the same x_.
    """
    if projection == "2d":
        data_.plot.scatter(
            "S1", "S2", c="difference_detected", cmap="viridis", zorder=10
        )
    elif projection == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(data_["S1"], data["S2"], data["difference_detected"])

        if estimator is not None:
            xs, ys = np.mgrid[0:5:0.2, 0:5:0.2]  # type: ignore

            zs = estimator.predict(np.column_stack((xs.ravel(), ys.ravel())))

            ax.plot_surface(xs, ys, zs.reshape(xs.shape), alpha=0.5)

    if label is not None:
        plt.title(label)

    if show_results:
        plt.show()

    return


# %% Load the data


data = pd.read_csv("example/studies/Weber/experimentation/experiment_0_data.csv")
show_results = partial(show_results_complete, data_=data, projection="3d")
show_results()
show_results(label="input data")

X = data[["S1", "S2"]]
y = data["difference_detected"]


# %% Fit first using a super-simple linear regression

first_order_linear_estimator = LinearRegression()
first_order_linear_estimator.fit(X, y)

show_results(estimator=first_order_linear_estimator, label="1st order linear")

# %% Fit using a 0-3 order polynomial, getting the best fit for the data.
polynomial_estimator = GridSearchCV(
    make_pipeline(PolynomialFeatures(), LinearRegression(fit_intercept=False)),
    param_grid=dict(polynomialfeatures__degree=range(4)),
)
polynomial_estimator.fit(X, y)

show_results(estimator=polynomial_estimator, label="[0th-3rd]-order linear")

# %% Fit using DARTS
darts_estimator = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        DARTS(
            batch_size=5,
            arch_updates_per_epoch=20,
            param_updates_per_epoch=20,
            max_epochs=50,
            output_type=ValueType.PROBABILITY,
        ),
    ),
    param_grid=dict(darts__num_graph_nodes=range(1, 5)),
)
darts_estimator.fit(X, y)

show_results(estimator=darts_estimator, label="DARTS")
print(darts_estimator.best_params_)

# %% Fit using Multilayer-Perceptron
mlp_estimator = MLPRegressor()
mlp_estimator.fit(X, y)

show_results(estimator=mlp_estimator, label="Multilayer-Perceptron")

# %%
show_results(
    estimators={
        "1st-order linear": first_order_linear_estimator,
        "[0th-5th]-order linear": polynomial_estimator,
        "DARTS": darts_estimator,
        "Multi-Layer-Perceptron": mlp_estimator,
    }
)
