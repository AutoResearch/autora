import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from autora.experimentalist.grid import grid_pool
from autora.state import StandardState, estimator_on_state, on_state
from autora.variable import Variable, VariableCollection

rng = np.random.default_rng()


def initial_state(_):
    state = StandardState(
        variables=VariableCollection(
            independent_variables=[
                Variable(name="x", allowed_values=np.linspace(-10, +10, 1001))
            ],
            dependent_variables=[Variable(name="y")],
            covariates=[],
        ),
        conditions=None,
        experiment_data=pd.DataFrame({"x": [], "y": []}),
        models=[],
    )
    return state


@on_state(output=["conditions"])
def experimentalist(variables):
    conditions: pd.DataFrame = grid_pool(variables)
    selected_conditions = conditions.sample(10, random_state=rng)
    return selected_conditions


coefs = [2.0, 3.0, 1.0]
noise_std = 10.0


def ground_truth(x, coefs_=coefs):
    return coefs_[0] * x**2.0 + coefs_[1] * x + coefs_[2]


@on_state(output=["experiment_data"])
def experiment_runner(conditions, coefs_=coefs, noise_std_=noise_std, rng=rng):
    experiment_data = conditions.assign(
        y=(
            ground_truth(conditions["x"], coefs_=coefs_)
            + rng.normal(0.0, noise_std_, size=conditions["x"].shape)
        )
    )
    return experiment_data


theorist = estimator_on_state(
    GridSearchCV(
        make_pipeline(PolynomialFeatures(), LinearRegression()),
        param_grid={"polynomialfeatures__degree": [0, 1, 2, 3, 4]},
        scoring="r2",
    )
)
