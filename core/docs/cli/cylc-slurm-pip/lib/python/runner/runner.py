import pandas as pd
from sklearn.linear_model import LinearRegression

from autora.experimentalist.grid import grid_pool
from autora.state import StandardState, estimator_on_state, on_state
from autora.variable import Variable, VariableCollection


def initial_state(_):
    state = StandardState(
        variables=VariableCollection(
            independent_variables=[Variable(name="x", allowed_values=range(100))],
            dependent_variables=[Variable(name="y")],
            covariates=[],
        ),
        conditions=None,
        experiment_data=pd.DataFrame({"x": [], "y": []}),
        models=[],
    )
    return state


experimentalist = on_state(grid_pool, output=["conditions"])

experiment_runner = on_state(
    lambda conditions: conditions.assign(y=2 * conditions["x"] + 0.5),
    output=["experiment_data"],
)

theorist = estimator_on_state(LinearRegression(fit_intercept=True))
