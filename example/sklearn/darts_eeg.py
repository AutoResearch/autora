"""
Example file which shows some simple curve fitting using DARTS and some other estimators.
"""

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aer.skl.darts import DARTS

matplotlib.use("module://backend_interagg")


# %% Define some helper functions

# Plot the results
def show_results_complete(data_, estimator=None, show=True):
    data_.plot.scatter(
        "EEG_Feature_A", "EEG_Feature_B", c="Diagnosis", cmap="viridis", zorder=10
    )

    if estimator is not None:
        ((xmin, xmax), (ymin, ymax)) = plt.xlim(), plt.ylim()
        n_steps = 200
        xstep, ystep = (xmax - xmin) / n_steps, (ymax - ymin) / n_steps
        gridx, gridy = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep]
        probs = logistic_estimator.predict_proba(
            np.column_stack([gridx.ravel(), gridy.ravel()])
        )[:, 1].reshape(gridx.shape)
        contour = plt.contourf(gridx, gridy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
        plt.colorbar(contour)
        # DecisionBoundaryDisplay.from_estimator(estimator, X_, ax=ax, alpha=0.5)

    if show:
        plt.show()


# %% Load the data

data = pd.read_csv("example/eeg/eeg_test_dataset.csv")
data["Diagnosis"] = pd.Categorical.from_codes(
    data["Diagnosis"], categories=["healthy", "diagnosed"]
)

show_results = partial(show_results_complete, data_=data)
show_results()

# %%

X = data[["EEG_Feature_A", "EEG_Feature_B"]]
y = data["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %% Fit first using a logistic regression

logistic_estimator = LogisticRegression()
logistic_estimator.fit(X_train, y_train)

show_results(estimator=logistic_estimator)

# %% Fit using DARTS
darts_estimator = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        DARTS(
            batch_size=20,
        ),
    ),
    param_grid=dict(darts__num_graph_nodes=range(1, 5)),
)
darts_estimator.fit(X_train, y_train)

show_results(estimators={"DARTS": darts_estimator})
print(darts_estimator.best_params_)

# %%
mlp_estimator = make_pipeline(StandardScaler(), MLPClassifier())
mlp_estimator.fit(X_train, y_train)

show_results(estimator=mlp_estimator)
