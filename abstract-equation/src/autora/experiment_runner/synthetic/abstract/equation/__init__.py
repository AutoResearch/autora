import warnings
from functools import partial
from itertools import product
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sympy import Expr, lambdify

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, VariableCollection


def equation_experiment(
    expression: Expr,
    X: List[IV],
    y: DV,
    name: str = "Equation Experiment",
    rename_output_columns: bool = True,
    random_state: Optional[int] = None,
):
    """

    A synthetic experiments that uses a sympy expression as ground truth.

    Sympy: https://www.sympy.org/en/index.html

    Args:
        expression: A sympy expression. The expression is interpreted as definition for a function
        X: The domain of independent variables
        y: The codomain of the dependent variables
        name: Name of the experiment
        random_state: Seed for random number generator
        rename_output_columns: If true, rename the columns of the output DataFrame based on the
            variable names in the expression.


    Examples:
        First we define an expression that will be interpreted as function. We need to define the
        symbols in sympy.
        >>> from sympy import symbols
        >>> x, y = symbols("x y")

        We also have to define the independent and dependent variables:
        >>> iv_x = IV(name='x', allowed_values=np.linspace(-10,10) ,value_range=(-10,10))
        >>> iv_y = IV(name='y', allowed_values=np.linspace(-10,10) ,value_range=(-10,10))
        >>> dv_z = DV(name='z')

        Now we can define an expression:
        >>> expr = x ** y

        Then we use this expression in our experiment
        >>> experiment = equation_experiment(expr, [iv_x, iv_y], dv_z, random_state=42)

        To run an experiment on some conditions, first we define those conditions as a pandas
        dataframe:
        >>> conditions = pd.DataFrame({'x':[1, 2, 3], 'y': [2, 3, 4]})
        >>> conditions
           x  y
        0  1  2
        1  2  3
        2  3  4

        Then to run the experiment, we pass that dataframe to the `.run` function:
        >>> experiment.run(conditions)
           x  y          z
        0  1  2   1.003047
        1  2  3   7.989600
        2  3  4  81.007505

        If the names the expression requires are not part of the dataframe, we get an error message:
        >>> experiment.run(
        ...     pd.DataFrame({'z':[1, 2, 2], 'x': [1, 2, 3]})
        ... )  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        Exception: Variables of expression x**y not found in columns of dataframe with columns
        Index(['z', 'x'], dtype='object')


        Each time an experiment is initialized with the same random_state, it should produce the
        same results:
        >>> experiment = equation_experiment(expr, [iv_x, iv_y], dv_z, random_state=42)
        >>> results42 = experiment.run(conditions)
        >>> results42
           x  y          z
        0  1  2   1.003047
        1  2  3   7.989600
        2  3  4  81.007505

        We can specify the random_state for a particular run to reproduce it:
        >>> results42_reproduced = experiment.run(conditions, random_state=42)
        >>> pd.DataFrame.equals(results42, results42_reproduced)
        True

        If we don't specify the random_state, it produces different values:
        >>> experiment.run(conditions)
           x  y          z
        0  1  2   1.009406
        1  2  3   7.980490
        2  3  4  80.986978

        An alternative input format for the experiment runner is a numpy array (not recommended):
        >>> experiment.run(np.array([[1, 1], [2, 2], [2, 3]]))
           x  y         z
        0  1  1  1.001278
        1  2  2  3.996838
        2  2  3  7.999832

        But we have to be careful with the order of the arguments in the runner. The arguments
        will be sorted alphabetically.
        In the following case the first entry of the numpy array is still x:
        >>> expr = y ** x
        >>> experiment.run(np.array([[1, 1], [2, 2] , [2, 3]]), random_state=42)
           x  y         z
        0  1  1  1.003047
        1  2  2  3.989600
        2  2  3  8.007505

    """

    params = dict(
        # Include all parameters here:
        expression=expression,
        name=name,
        random_state=random_state,
    )

    args = list(expression.free_symbols)
    args = sorted(args, key=lambda el: el.name)

    f_numpy = lambdify(args, expression, "numpy")

    # Define variables
    variables = VariableCollection(
        independent_variables=X,
        dependent_variables=[y],
    )
    if not set([el.name for el in variables.independent_variables]).issubset(
        set([str(a) for a in args])
    ):
        raise Exception(
            f"Independent variables {[iv.name for iv in X]} and symbols of the equation tree "
            f"{args} do not match."
        )

    # Define experiment runner
    rng = np.random.default_rng(random_state)

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise=0.01,
        random_state=None,
    ):
        """A function which simulates noisy observations."""

        if random_state is not None:
            rng_ = np.random.default_rng(random_state)
        else:
            rng_ = rng  # use the RNG from the outer scope

        x = conditions
        if isinstance(x, pd.DataFrame):
            x = x.copy()
            if not set([el.name for el in args]).issubset(x.columns):
                raise Exception(
                    f"Variables of expression {expression} "
                    f"not found in columns of dataframe with columns {x.columns}"
                )
            x_filtered = x[[el.name for el in args]]
            x_sorted = x_filtered.sort_index(axis=1)
            x_ = np.array(x_sorted)
        else:
            x_ = x
            warnings.warn(
                "Unnamed data is used. Arguments will be sorted alphabetically. "
                "Consider using a Pandas DataFrame with named columns for "
                "better clarity and ease of use.",
                category=RuntimeWarning,
            )

        out = f_numpy(*x_.T)
        out = out + rng_.normal(0, added_noise, size=out.shape)
        if isinstance(x, pd.DataFrame):
            _res = pd.DataFrame(x_, columns=x_sorted.columns)
            res = x
            for col in x_sorted.columns:
                res[col] = list(_res[col])
        else:
            if rename_output_columns:
                res = pd.DataFrame(x_, columns=[el.name for el in args])
            else:
                res = pd.DataFrame(x_.T)

        res[y.name] = out
        return res

    ground_truth = partial(run, added_noise_=0.0)
    """A function which simulates perfect observations"""

    def domain():
        """A function which returns all possible independent variable values as a grid."""
        iv_values = [iv.allowed_values for iv in variables.independent_variables[0]]
        X_combinations = product(*iv_values)
        X = np.array(list(X_combinations))
        return X

    def plotter(model=None):
        """A function which plots the ground truth and (optionally) a fitted model."""
        import matplotlib.pyplot as plt

        plt.figure()
        dom = domain()
        data = ground_truth(dom)

        y = data["observations"]
        x = data.drop("observations", axis=1)

        if x.shape[1] > 2:
            Exception(
                "No standard way to plot more then 2 independent variables implemented"
            )

        if x.shape[1] == 1:
            plt.plot(x, y, label="Ground Truth")
            if model is not None:
                plt.plot(x, model.predict(x), label="Fitted Model")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            x_ = x.iloc[:, 0]

            y_ = x.iloc[:, 1]
            z_ = y

            ax.scatter(x_, y_, z_, s=1, alpha=0.3, label="Ground Truth")
            if model is not None:
                z_m = model.predict(x)
                ax.scatter(x_, y_, z_m, s=1, alpha=0.5, label="Fitted Model")

        plt.legend()
        plt.title(name)
        plt.show()

    # The object which gets stored in the synthetic inventory
    collection = SyntheticExperimentCollection(
        name=name,
        description=equation_experiment.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=equation_experiment,
    )
    return collection
