"""
Objects for handling input and outputs from experimentalists, experiment runners and theorists.
"""

from __future__ import annotations

import copy
from functools import partial
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

from autora.controller.protocol.v1 import (
    Executor,
    ExecutorCollection,
    State,
    SupportsControllerState,
)
from autora.controller.state import resolve_state_params
from autora.experimentalist.pipeline import Pipeline


class OnlineExecutorCollection:
    """
    Runs experiment design, observation and theory generation in a single session.

    This object allows a user to specify
    - an experimentalist: a Pipeline
    - an experiment runner: some Callable and
    - a theorist: a scikit-learn-compatible estimator with a fit method

    ... and exposes methods to call these and update a CycleState object with new data.

    Examples:
        >>> from autora.experimentalist.pipeline import Pipeline
        >>> from sklearn.linear_model import LinearRegression
        >>> experimentalist_pipeline_ = Pipeline([('p', (1, 2))])
        >>> def experiment_runner_(x):
        ...     return 2 * x + 1
        >>> theorist_estimator_ = LinearRegression()
        >>> c = OnlineExecutorCollection(
        ...     experimentalist_pipeline=experimentalist_pipeline_,
        ...     theorist_estimator=theorist_estimator_,
        ...     experiment_runner_callable=experiment_runner_
        ... )
        >>> c  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        OnlineExecutorCollection(experimentalist_pipeline=Pipeline(steps=[('p', (1, 2))],
        params={}), experiment_runner_callable=<function experiment_runner_ at 0x...>,
        theorist_estimator=LinearRegression())

        We can access the collection as a mapping:
        >>> c["experimentalist_pipeline"]
        Pipeline(steps=[('p', (1, 2))], params={})

        ... or using the attributes directly:
        >>> c.experimentalist_pipeline
        Pipeline(steps=[('p', (1, 2))], params={})

        Updating the pipeline functions
    """

    def __init__(
        self,
        experimentalist_pipeline: Pipeline,
        experiment_runner_callable: Callable,
        theorist_estimator: BaseEstimator,
    ):
        self.experimentalist_pipeline = experimentalist_pipeline
        self.experiment_runner_callable = experiment_runner_callable
        self.theorist_estimator = theorist_estimator

    def __getitem__(self, item):
        """Mapping interface."""
        return getattr(self, item)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"experimentalist_pipeline={self.experimentalist_pipeline}, "
            f"experiment_runner_callable={self.experiment_runner_callable}, "
            f"theorist_estimator={self.theorist_estimator}"
            f")"
        )

    def experimentalist(
        self, state: SupportsControllerState
    ) -> SupportsControllerState:
        """Interface for running the experimentalist pipeline."""
        new_state = experimentalist_wrapper(state, self.experiment_runner_callable)
        return new_state

    def experiment_runner(
        self, state: SupportsControllerState
    ) -> SupportsControllerState:
        """Interface for running the experiment runner callable"""
        new_state = experiment_runner_wrapper(state, self.experiment_runner_callable)
        return new_state

    def theorist(self, state: SupportsControllerState) -> SupportsControllerState:
        """Interface for running the theorist estimator."""
        new_state = theorist_wrapper(state, self.theorist_estimator)
        return new_state

    def full_cycle(self, state: SupportsControllerState) -> SupportsControllerState:
        """
        Executes the experimentalist, experiment runner and theorist on the given state.

        Returns: A list of new results
        """
        experimentalist_result = self.experimentalist(state)
        experiment_runner_result = self.experiment_runner(experimentalist_result)
        theorist_result = self.theorist(experiment_runner_result)
        return theorist_result


def experimentalist_wrapper(
    state: SupportsControllerState, pipeline: Pipeline
) -> SupportsControllerState:
    """Interface for running the experimentalist pipeline."""
    params = resolve_state_params(state).get("experimentalist", dict())
    new_conditions = pipeline(**params)

    assert isinstance(new_conditions, Iterable)
    # If the pipeline gives us an iterable, we need to make it into a concrete array.
    # We can't move this logic to the Pipeline, because the pipeline doesn't know whether
    # it's within another pipeline and whether it should convert the iterable to a
    # concrete array.
    new_conditions_values = list(new_conditions)
    new_conditions_array = np.array(new_conditions_values)

    assert isinstance(new_conditions_array, np.ndarray)  # Check the object is bounded
    new_state = state.update(conditions=[new_conditions_array])
    return new_state


def experiment_runner_wrapper(
    state: SupportsControllerState, callable: Callable
) -> SupportsControllerState:
    """Interface for running the experiment runner callable"""
    params = resolve_state_params(state).get("experiment_runner", dict())
    x = state.conditions[-1]
    y = callable(x, **params)
    new_observations = np.column_stack([x, y])
    new_state = state.update(observations=[new_observations])
    return new_state


def theorist_wrapper(state: State, estimator: BaseEstimator) -> State:
    params = resolve_state_params(state).get("theorist", dict())
    metadata = state.metadata
    observations = state.observations
    all_observations = np.row_stack(observations)
    n_xs = len(metadata.independent_variables)
    x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
    if y.shape[1] == 1:
        y = y.ravel()
    new_theorist = copy.deepcopy(estimator)
    new_theorist.fit(x, y, **params)
    new_state = state.update(theories=[new_theorist])
    return new_state


def full_cycle_wrapper(
    state: State,
    experimentalist_pipeline: Pipeline,
    experiment_runner_callable: Callable,
    theorist_estimator: BaseEstimator,
):
    experimentalist_result = experimentalist_wrapper(state, experimentalist_pipeline)
    experiment_runner_result = experiment_runner_wrapper(
        experimentalist_result, experiment_runner_callable
    )
    theorist_result = theorist_wrapper(experiment_runner_result, theorist_estimator)
    return theorist_result


def make_online_executor(
    kind: Literal["experimentalist", "experiment_runner", "theorist"],
    core: Union[Pipeline, Callable, BaseEstimator],
) -> Executor:
    """

    Args:
        kind: a string specifying the kind of function (and thus the correct wrapper to use)
        core: the object to wrap – "experimentalist": a Pipeline, "experiment_runner": a
            Callable, "theorist": a BaseEstimator

    Returns: a curried function which will run the kind of AER step requested

    """
    if kind == "experimentalist":
        assert isinstance(core, Pipeline)
        curried_function = partial(experimentalist_wrapper, pipeline=core)
    elif kind == "experiment_runner":
        assert isinstance(core, Callable)
        curried_function = partial(experiment_runner_wrapper, callable=core)
    elif kind == "theorist":
        assert isinstance(core, BaseEstimator)
        curried_function = partial(theorist_wrapper, estimator=core)
    else:
        raise NotImplementedError(
            f"{kind=} is not implemented for executor definitions."
        )
    return curried_function


def make_executor_collection(
    x: Iterable[
        Tuple[
            str,
            Literal["experimentalist", "experiment_runner", "theorist"],
            Union[Pipeline, Callable, BaseEstimator],
        ]
    ]
) -> ExecutorCollection:
    """

    Make an executor collection using experimentalists, experiment_runners and theorists.

    Args:
        x:

    Returns:

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> make_executor_collection([("t", "theorist", LinearRegression())]) # doctest: +ELLIPSIS
        {'t': functools.partial(<function theorist_wrapper at 0x...>, estimator=LinearRegression())}

        >>> make_executor_collection([("er", "experiment_runner", lambda x_: x_ + 1)]
        ... ) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'er': functools.partial(<function experiment_runner_wrapper at 0x...>, callable=<function
        <lambda> at 0x...>)}
    """
    c: ExecutorCollection = {}
    for name, kind, core in x:
        c[name] = make_online_executor(kind, core)

    return c


def make_default_executor_collection(
    experimentalist_pipeline: Pipeline,
    experiment_runner_callable: Callable,
    theorist_estimator: BaseEstimator,
) -> ExecutorCollection:
    """
    Make the default AER executor collection.

    Args:
        experimentalist_pipeline: an experimentalist Pipeline to be wrapped
        experiment_runner_callable: an experiment runner function to be wrapped
        theorist_estimator: a scikit learn-compatible estimator to be wrapped

    Returns: A dictionary with keys "experimentalist", "experiment_runner", "theorist" and
        "full_cycle", with values which are Callables.


    Examples:
        >>> from autora.experimentalist.pipeline import Pipeline
        >>> from sklearn.linear_model import LinearRegression
        >>> experimentalist_pipeline_ = Pipeline([('p', (1, 2))])
        >>> def experiment_runner_(x):
        ...     return 2 * x + 1
        >>> theorist_estimator_ = LinearRegression()
        >>> c = make_default_executor_collection(
        ...     experimentalist_pipeline=experimentalist_pipeline_,
        ...     theorist_estimator=theorist_estimator_,
        ...     experiment_runner_callable=experiment_runner_
        ... )
        >>> c  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        mappingproxy({'experimentalist': functools.partial(...),
         'experiment_runner': functools.partial(...),
         'theorist': functools.partial(...),
         'full_cycle': functools.partial(...)})

        We can access the collection as a mapping:
        >>> c["experimentalist"]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        functools.partial(<function experimentalist_wrapper at 0x...>,
        pipeline=Pipeline(steps=[('p', (1, 2))], params={}))

        ...
        >>> c["experiment_runner"]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        functools.partial(<function experiment_runner_wrapper at 0x...>,
                          callable=<function experiment_runner_ at 0x...>)

        ...
        >>> c["theorist"]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        functools.partial(<function theorist_wrapper at 0x...>,
                          estimator=LinearRegression())

        ...
        >>> c["full_cycle"]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        functools.partial(<function full_cycle_wrapper at 0x...>,
            experimentalist_pipeline=Pipeline(steps=[('p', (1, 2))], params={}),
            experiment_runner_callable=<function experiment_runner_ at 0x...>,
            theorist_estimator=LinearRegression())

        You cannot update the collection. To replace a value, create a new collection.
        >>> other_theorist = LinearRegression(fit_intercept=False)
        >>> c["theorist"] = partial(theorist_wrapper, estimator=other_theorist)
        Traceback (most recent call last):
        ...
        TypeError: 'mappingproxy' object does not support item assignment

        (Updating the collection is restricted because the "full_cycle" depends on all the
        input functions directly, and doesn't reuse the wrapped functions created for the
        experimentalist, experiment_runner and theorist.)

    """

    c = make_executor_collection(
        [
            ("experimentalist", "experimentalist", experimentalist_pipeline),
            ("experiment_runner", "experiment_runner", experiment_runner_callable),
            ("theorist", "theorist", theorist_estimator),
        ]
    )

    c["full_cycle"] = partial(
        full_cycle_wrapper,
        experimentalist_pipeline=experimentalist_pipeline,
        experiment_runner_callable=experiment_runner_callable,
        theorist_estimator=theorist_estimator,
    )

    return MappingProxyType(c)
