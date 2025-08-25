"""
A synthetic experiment that runs a linear mixed model.

Examples:
    >>> from autora.experiment_runner.synthetic.abstract.lmm import (
    ...     lmm_experiment
    ... )

    >>> formula = 'rt ~ 1'
    >>> fixed_effects = {'Intercept': 1.5}
    >>> experiment = lmm_experiment(formula=formula,fixed_effects=fixed_effects)
    >>> conditions = pd.DataFrame({
    ...     'x1':np.linspace(0, 1, 5)
    ... })
    >>> experiment.ground_truth(conditions=conditions)
         x1   rt
    0  0.00  1.5
    1  0.25  1.5
    2  0.50  1.5
    3  0.75  1.5
    4  1.00  1.5

    >>> formula = 'rt ~ 1 + x1'
    >>> fixed_effects = {'Intercept': 1., 'x1': 2.}
    >>> experiment = lmm_experiment(formula=formula,fixed_effects=fixed_effects)
    >>> experiment.ground_truth(conditions=conditions)
         x1   rt
    0  0.00  1.0
    1  0.25  1.5
    2  0.50  2.0
    3  0.75  2.5
    4  1.00  3.0

    >>> formula_1 = 'rt ~ 1 + x1'
    >>> fixed_effects_1 = {'Intercept': 0., 'x1': 2.}
    >>> experiment_1 = lmm_experiment(formula=formula_1,fixed_effects=fixed_effects_1)
    >>> formula_2 = 'rt ~ x1'
    >>> fixed_effects_2 = {'x1': 2.}
    >>> experiment_2 = lmm_experiment(formula=formula_2,fixed_effects=fixed_effects_2)
    >>> experiment_1.ground_truth(conditions=conditions) ==\
experiment_2.ground_truth(conditions=conditions)
         x1    rt
    0  True  True
    1  True  True
    2  True  True
    3  True  True
    4  True  True

    >>> formula = 'rt ~ 1 + (1|subject) + x1'
    >>> fixed_effects = {'Intercept': 1, 'x1': 2}
    >>> random_effects = {'subject': {'Intercept': .1}}
    >>> experiment = lmm_experiment(formula=formula,
    ...                             fixed_effects=fixed_effects,
    ...                             random_effects=random_effects)
    >>> conditions_1 = pd.DataFrame({
    ...     'x1':np.linspace(0, 1, 3),
    ...     'subject': np.repeat(1, 3)
    ... })
    >>> conditions_2 = pd.DataFrame({
    ...     'x1':np.linspace(0, 1, 3),
    ...     'subject': np.repeat(2, 3)
    ... })
    >>> conditions = pd.concat([conditions_1, conditions_2])
    >>> conditions
        x1  subject
    0  0.0        1
    1  0.5        1
    2  1.0        1
    0  0.0        2
    1  0.5        2
    2  1.0        2
    >>> experiment.ground_truth(conditions=conditions,random_state=42)
        x1  subject        rt
    0  0.0        1  1.030472
    1  0.5        1  2.030472
    2  1.0        1  3.030472
    0  0.0        2  0.896002
    1  0.5        2  1.896002
    2  1.0        2  2.896002

    >>> formula = 'rt ~ (x1|subject)'
    >>> random_effects = {'subject': {'x1': .1}}
    >>> experiment = lmm_experiment(formula=formula,random_effects=random_effects)
    >>> experiment.ground_truth(conditions=conditions,random_state=42)
        x1  subject        rt
    0  0.0        1  0.000000
    1  0.5        1  0.015236
    2  1.0        1  0.030472
    0  0.0        2  0.000000
    1  0.5        2 -0.051999
    2  1.0        2 -0.103998

    >>> formula = 'rt ~ (x1|subject) + x1'
    >>> fixed_effects = {'x1': 1.}
    >>> random_effects = {'subject': {'x1': .01}}
    >>> experiment = lmm_experiment(formula=formula,
    ...                             fixed_effects=fixed_effects,
    ...                             random_effects=random_effects)
    >>> experiment.ground_truth(conditions=conditions,random_state=42)
        x1  subject        rt
    0  0.0        1  0.000000
    1  0.5        1  0.501524
    2  1.0        1  1.003047
    0  0.0        2  0.000000
    1  0.5        2  0.494800
    2  1.0        2  0.989600

    >>> formula = 'y ~ x1 + x2 + (1 + x1|subject) + (x2|group)'
    >>> fixed_effects = {'Intercept': 1.5, 'x1': 2.0, 'x2': -1.2}
    >>> random_effects = {
    ...        'subject': {'1': 0.5, 'x1': 0.3},
    ...        'group': {'x2': 0.4}
    ...    }
    >>> experiment = lmm_experiment(formula=formula,
    ...                             fixed_effects=fixed_effects,
    ...                             random_effects=random_effects)
    >>> n_samples = 10
    >>> rng = np.random.default_rng(0)
    >>> conditions = pd.DataFrame({
    ...        'x1': rng.normal(0, 1, n_samples),
    ...        'x2': rng.normal(0, 1, n_samples),
    ...        'subject': rng.choice(['A', 'B', 'C', 'D'], n_samples),
    ...        'group': rng.choice(['E', 'F', 'G', 'H'], n_samples)
    ...    })
    >>> experiment.ground_truth(conditions=conditions, random_state=42)
             x1        x2 subject group         y
    0  0.125730 -0.623274       B     H  2.502995
    1 -0.132105  0.041326       A     F  1.258294
    2  0.640423 -2.325031       A     F  5.490146
    3  0.104900 -0.218792       A     H  1.899763
    4 -0.535669 -1.245911       A     H  2.173576
    5  0.361595 -0.732267       C     H  2.923207
    6  1.304000 -0.544259       C     F  4.320545
    7  0.947081 -0.316300       C     G  3.405867
    8 -0.703735  0.411631       B     H -0.578950
    9 -1.265421  1.042513       C     G -1.794523

    >>> experiment.run(conditions=conditions, added_noise=.1, random_state=42)
             x1        x2 subject group         y
    0  0.125730 -0.623274       B     H  2.417691
    1 -0.132105  0.041326       A     F  1.346234
    2  0.640423 -2.325031       A     F  5.567925
    3  0.104900 -0.218792       A     H  1.906366
    4 -0.535669 -1.245911       A     H  2.286300
    5  0.361595 -0.732267       C     H  2.969958
    6  1.304000 -0.544259       C     F  4.234616
    7  0.947081 -0.316300       C     G  3.442742
    8 -0.703735  0.411631       B     H -0.674839
    9 -1.265421  1.042513       C     G -1.706678

"""

import re
from functools import partial
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, VariableCollection


def lmm_experiment(
    # Add any configurable parameters with their defaults here:
    formula: str,
    fixed_effects: Optional[dict] = None,
    random_effects: Optional[dict] = None,
    X: Optional[Sequence[IV]] = None,
    random_state: Optional[int] = None,
    name: str = "Linear Mixed Model Experiment",
):
    """
    A linear mixed model synthetic experiments.

    Parameters:
        name: name of the experiment
        formula: formula of the linear mixed model (similar to lmer package in R)
        fixed_effects: dictionary describing the fixed effects (Intercept and slopes)
        random_effects: nested dictionary describing the random effects of slopes and intercept.
            These are standard deviasions in a normal distribution with a mean of zero.
        X: Independent variable descriptions. Used to add allowed values
    """

    if not fixed_effects:
        fixed_effects = {}
    if not random_effects:
        random_effects = {}

    params = dict(
        # Include all parameters here:
        name=name,
        formula=formula,
        fixed_effects=fixed_effects,
        random_effects=random_effects,
    )

    dependent, fixed_variables, random_variables = _extract_variable_names(formula)

    dependent = DV(name=dependent)
    if not X:
        independent = [IV(name=iv) for iv in fixed_variables + random_variables]
    else:
        if set([x.name for x in X]) != set(fixed_variables + random_variables):
            raise Exception(
                "Variable names in formula don't match given variable names"
            )
        independent = X

    variables = VariableCollection(
        independent_variables=independent,
        dependent_variables=[dependent],
    )

    rng = np.random.default_rng(random_state)

    # Define experiment runner
    def run(
        conditions: pd.DataFrame,
        added_noise=0.01,
        random_state=None,
    ):
        """A function which simulates noisy observations."""
        if random_state is not None:
            rng_ = np.random.default_rng(random_state)
        else:
            rng_ = rng  # use the RNG from the outer scope

        dependent_var, rhs = formula.split("~")
        dependent_var = dependent_var.strip()
        fixed_vars = fixed_variables

        # Check for the presence of an intercept in the formula
        has_intercept = (
            True if "1" in fixed_effects or re.search(r"\b0\b", rhs) is None else False
        )

        if not isinstance(conditions, pd.DataFrame):
            _conditions = np.array(conditions)
            _conditions = pd.DataFrame(_conditions)
            _conditions.columns = [iv.name for iv in variables.independent_variables]
        else:
            _conditions = conditions
        experiment_data = _conditions.copy()

        # Initialize the dependent variable
        experiment_data[dependent_var] = (
            fixed_effects.get("Intercept", 0) if has_intercept else 0
        )

        # Add fixed effects
        for var in fixed_vars:
            if var in experiment_data.columns:
                experiment_data[dependent_var] += (
                    fixed_effects.get(var, 0) * experiment_data[var]
                )

        # Process each random effect term
        random_effect_terms = re.findall(r"\((.+?)\|(.+?)\)", formula)
        for term in random_effect_terms:
            random_effects_, group_var = term
            group_var = group_var.strip()

            # Ensure the group_var is in the data
            if group_var not in experiment_data.columns:
                raise ValueError(f"Group variable '{group_var}' not found in the data")

            # Process each part of the random effect (intercept and slopes)
            for part in random_effects_.split("+"):
                part = "Intercept" if part == "1" else part
                part = part.strip()
                std_dev = random_effects[group_var].get(part, 0.5)
                random_effect_values = {
                    group: rng_.normal(0, std_dev)
                    for group in experiment_data[group_var].unique()
                }
                if part == "Intercept":  # Random intercept
                    if has_intercept:
                        experiment_data[dependent_var] += experiment_data[
                            group_var
                        ].map(random_effect_values)
                else:  # Random slopes
                    if part in experiment_data.columns:
                        experiment_data[dependent_var] += (
                            experiment_data[group_var].map(random_effect_values)
                            * experiment_data[part]
                        )

        # Add noise
        experiment_data[dependent_var] += rng_.normal(
            0, added_noise, len(experiment_data)
        )

        return experiment_data

    ground_truth = partial(run, added_noise=0.0)
    """A function which simulates perfect observations.
    This still uses random values for random effects."""

    def domain():
        """A function which returns all possible independent variable values as a 2D array."""
        x = variables.independent_variables[0].allowed_values.reshape(-1, 1)
        return x

    def plotter(model=None):
        """A function which plots the ground truth and (optionally) a fitted model."""
        import matplotlib.pyplot as plt

        plt.figure()
        dom = domain()
        data = ground_truth(dom)

        y = data[dependent.name]
        x = data.drop(dependent.name, axis=1)

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
        description=lmm_experiment.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=lmm_experiment,
    )
    return collection


def _extract_variable_names(formula):
    """
    Extract fixed and random effects from a linear mixed model formula.

    Parameters:
    formula (str): Formula specifying the model, e.g., 'y ~ x1 + x2 + (1 + x1|group) + (x2|subject)'

    Returns:
    tuple of (list, list): A tuple containing two lists - one for fixed effects and another for
    random effects.
    Examples:
        >>> formula_1 = 'y ~ x1 + x2 + (1 + x1|group) + (x2|subject)'
        >>> _extract_variable_names(formula_1)
        ('y', ['x1', 'x2'], ['group', 'subject'])

        >>> formula_2 = 'rt ~ x_1 + (x_2|group)'
        >>> _extract_variable_names(formula_2)
        ('rt', ['x_1', 'x_2'], ['group'])

        >>> formula_3 = 'RT ~ 1'
        >>> _extract_variable_names(formula_3)
        ('RT', [], [])


    """
    # Extract the right-hand side of the formula
    dependent, rhs = formula.split("~")
    dependent = dependent.strip()

    fixed_effects = re.findall(
        r"[a-z]\w*(?![^\(]*\))", rhs
    )  # Matches variables outside parentheses
    random_effects = re.findall(
        r"\(([^\|]+)\|([^\)]+)\)", rhs
    )  # Matches random effects groups

    # Include variables from random effects in fixed effects and make unique
    for reffect in random_effects:
        fixed_effects.extend(reffect[0].replace("1 + ", "").split("+"))

    # Removing duplicates and stripping whitespaces
    fixed_effects = sorted(list(set([effect.strip() for effect in fixed_effects])))
    random_groups = sorted(
        list(set([reffect[1].strip() for reffect in random_effects]))
    )

    return dependent, fixed_effects, random_groups
