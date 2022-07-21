"""
Example file which shows some simple curve fitting using DARTS and some other estimators.
"""

from collections import Iterable
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from aer.skl.darts import DARTS

matplotlib.use("module://backend_interagg")


# %% Define some helper functions


def show_results_complete(
    estimators: dict = dict(), x_=np.zeros(0), y_=np.zeros(0), show_results=True
):
    """
    Function to plot input data (x_, y_) and the predictions of an estimator for the same x_.
    """
    plt.scatter(x_, y_, s=1, label="Input data")
    for (label, estimator) in estimators.items():
        plt.plot(x_, estimator.predict(x_.reshape(-1, 1)), label=label)
    plt.legend()
    if show_results:
        plt.show()


def format_polynomial_coefficients(c: Iterable, iv="x", dv="y"):
    """
    Format a polynomial equation of the form: y = c[i] x ** i + ... c[1] x + c[0]

    Args:
        c: c[i] ordered from smallest x**0 to largest x**[i=n]
        iv: name to use for independent variables
        dv: name to use for dependent variable

    Returns:
        formatted_string: like y = -3.02 x**2 + 6.10 x + 5.00

    """

    def format_exponent(i):
        if i == 0:
            return ""
        elif i == 1:
            return iv
        else:
            return f"{iv}**{i}"

    ivs = " + ".join(
        reversed(  # order from largest to smallest exponent: c[2] x**2 + c[1] x + c[0]
            [f"{c:.2f} " + format_exponent(i) for i, c in enumerate(c)]
        )
    )

    formatted_string = f"Best fit: {dv} = {ivs}"
    return formatted_string


# %% Generate the data


def func(x_):
    """
    Function dominated by a 2nd-order polynomial, but with a small 3rd-order component.
    """
    y_ = 0.1 * (x_**3) - 9.81 * (x_**2) + 6 * x_ + 5.0
    return y_


# Sample x between 0 and 1, including some noise.
n_samples = 1000
noise = 1 / 10
x = np.linspace(0, 1, n_samples)
y = func(x) + np.random.default_rng(42).normal(scale=noise, size=n_samples)

# Reshape the input-values x to match the expected format from sklearn.
x = x.reshape(-1, 1)

# Initialize a function which can plot the results
show_results = partial(show_results_complete, x_=x, y_=y)

show_results()

# %% Fit first using a super-simple linear regression

first_order_linear_estimator = LinearRegression()
first_order_linear_estimator.fit(x, y)

# Report the results

print(
    format_polynomial_coefficients(
        [
            first_order_linear_estimator.coef_.item(),
            first_order_linear_estimator.intercept_,
        ]
    )
)

show_results({"1st-order Linear": first_order_linear_estimator})

# %% Fit using a 0-5 order polynomial, getting the best fit for the data.
polynomial_estimator = GridSearchCV(
    make_pipeline(PolynomialFeatures(), LinearRegression(fit_intercept=False)),
    param_grid=dict(polynomialfeatures__degree=range(6)),
)
polynomial_estimator.fit(x, y)

print(
    format_polynomial_coefficients(
        np.nditer(
            polynomial_estimator.best_estimator_.named_steps["linearregression"].coef_
        )
    )
)

show_results(estimators={"[0th-5th]-order linear": polynomial_estimator})

# %% Fit using DARTS
darts_estimator = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        DARTS(batch_size=20),
    ),
    param_grid=dict(darts__num_graph_nodes=range(1, 5)),
)
darts_estimator.fit(x, y)

show_results(estimators={"DARTS": darts_estimator})
print(darts_estimator.best_params_)

# %%
mlp_estimator = make_pipeline(
    StandardScaler(), PolynomialFeatures(degree=(0, 5)), MLPRegressor()
)
mlp_estimator.fit(x, y)

show_results(estimators={"Multi-Layer-Perceptron": mlp_estimator})

# %%
show_results(
    estimators={
        "1st-order linear": first_order_linear_estimator,
        "[0th-5th]-order linear": polynomial_estimator,
        "DARTS": darts_estimator,
        "Multi-Layer-Perceptron": mlp_estimator,
    }
)
