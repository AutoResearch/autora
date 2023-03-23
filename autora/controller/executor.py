"""
Objects for handling input and outputs from experimentalists, experiment runners and theorists.
"""

from __future__ import annotations

import copy
import logging
from functools import partial
from types import MappingProxyType
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

from autora.controller.protocol import SupportsControllerState
from autora.controller.state import resolve_state_params
from autora.experimentalist.pipeline import Pipeline

_logger = logging.getLogger(__name__)


def experimentalist_wrapper(
    state: SupportsControllerState, pipeline: Pipeline, params: Dict
) -> SupportsControllerState:
    """Interface for running the experimentalist pipeline."""
    params_ = resolve_state_params(params, state)
    new_conditions = pipeline(**params_)

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
    state: SupportsControllerState, callable: Callable, params: Dict
) -> SupportsControllerState:
    """Interface for running the experiment runner callable."""
    params_ = resolve_state_params(params, state)
    x = state.conditions[-1]
    y = callable(x, **params_)
    new_observations = np.column_stack([x, y])
    new_state = state.update(observations=[new_observations])
    return new_state


def theorist_wrapper(
    state: SupportsControllerState, estimator: BaseEstimator, params: Dict
) -> SupportsControllerState:
    """Interface for running the theorist estimator given some State."""
    params_ = resolve_state_params(params, state)
    metadata = state.metadata
    observations = state.observations
    all_observations = np.row_stack(observations)
    n_xs = len(metadata.independent_variables)
    x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
    if y.shape[1] == 1:
        y = y.ravel()
    new_theorist = copy.deepcopy(estimator)
    new_theorist.fit(x, y, **params_)
    new_state = state.update(theories=[new_theorist])
    return new_state


def full_cycle_wrapper(
    state: SupportsControllerState,
    experimentalist_pipeline: Pipeline,
    experiment_runner_callable: Callable,
    theorist_estimator: BaseEstimator,
    params: Dict,
) -> SupportsControllerState:
    """Interface for running the full AER cycle."""
    experimentalist_params = params.get("experimentalist", {})
    experimentalist_result = experimentalist_wrapper(
        state, experimentalist_pipeline, experimentalist_params
    )
    experiment_runner_params = params.get("experiment_runner", {})
    experiment_runner_result = experiment_runner_wrapper(
        experimentalist_result, experiment_runner_callable, experiment_runner_params
    )
    theorist_params = params.get("theorist", {})
    theorist_result = theorist_wrapper(
        experiment_runner_result, theorist_estimator, theorist_params
    )
    return theorist_result


def no_op(state, params):
    """
    An Executor which has no effect on the state.

    Examples:
         >>> from autora.controller.state import Snapshot
         >>> s = Snapshot()
         >>> s_returned = no_op(s, {})
         >>> assert s_returned is s
    """
    _logger.warning("You called a `no_op` Executor. Returning the state unchanged.")
    return state


def make_online_executor(
    kind: Literal["experimentalist", "experiment_runner", "theorist"],
    core: Optional[Union[Pipeline, Callable, BaseEstimator]] = None,
):
    """

    Args:
        kind: a string specifying the kind of function (and thus the correct wrapper to use)
        core: the object to wrap – "experimentalist": a Pipeline, "experiment_runner": a
            Callable, "theorist": a BaseEstimator

    Returns: a curried function which will run the kind of AER step requested

    Examples:
        Initializing executors which are understood:
        >>> make_online_executor("experiment_runner", lambda x: x + 1
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        functools.partial(<function experiment_runner_wrapper at 0x...>,
                          callable=<function <lambda> at 0x...>)

        >>> make_online_executor("not_allowed_kind", lambda x: x + 1)
        Traceback (most recent call last):
        ...
        NotImplementedError: kind='not_allowed_kind' is not implemented for executor definitions.

    """
    if core is None:
        curried_function = no_op
    elif kind == "experimentalist":
        assert isinstance(core, Pipeline)
        curried_function = partial(experimentalist_wrapper, pipeline=core)
    elif kind == "experiment_runner":
        assert callable(core)
        curried_function = partial(experiment_runner_wrapper, callable=core)
    elif kind == "theorist":
        assert isinstance(core, BaseEstimator)
        curried_function = partial(theorist_wrapper, estimator=core)
    else:
        raise NotImplementedError(
            f"{kind=} is not implemented for executor definitions."
        )
    return curried_function


def make_online_executor_collection(
    x: Iterable[
        Tuple[
            str,
            Literal["experimentalist", "experiment_runner", "theorist"],
            Union[Pipeline, Callable, BaseEstimator],
        ]
    ]
):
    """
    Make an executor collection using experimentalists, experiment_runners and theorists.

    Args:
        x:

    Returns:

    Examples:
        We can create an executor collection with one theorist:
        >>> from sklearn.linear_model import LinearRegression
        >>> make_online_executor_collection([("t", "theorist", LinearRegression())]
        ... ) # doctest: +ELLIPSIS
        {'t': functools.partial(<function theorist_wrapper at 0x...>, estimator=LinearRegression())}

        ... or with two different theorists (e.g., if a Planner had several options)
        >>> make_online_executor_collection([
        ...     ("t0", "theorist", LinearRegression(fit_intercept=False)),
        ...     ("t1", "theorist", LinearRegression(fit_intercept=True))
        ... ]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'t0': functools.partial(<function theorist_wrapper at 0x...>,
                                 estimator=LinearRegression(fit_intercept=False)),
        't1': functools.partial(<function theorist_wrapper at 0x...>,
                                estimator=LinearRegression())}

        The same applies for experiment runners:
        >>> make_online_executor_collection([("er", "experiment_runner", lambda x_: x_ + 1)]
        ... ) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'er': functools.partial(<function experiment_runner_wrapper at 0x...>,
                                 callable=<function <lambda> at 0x...>)}

        ... and experimentalists:
        >>> from autora.experimentalist.pipeline import make_pipeline
        >>> make_online_executor_collection([("ex", "experimentalist", make_pipeline([(1,2,3)]))]
        ... ) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'ex': functools.partial(<function experimentalist_wrapper at 0x...>,
                                 pipeline=Pipeline(steps=[('step', (1, 2, 3))], params={}))}

    """
    c = {}
    for name, kind, core in x:
        c[name] = make_online_executor(kind, core)

    return c


def make_default_online_executor_collection(
    experimentalist_pipeline: Optional[Pipeline] = None,
    experiment_runner_callable: Optional[Callable] = None,
    theorist_estimator: Optional[BaseEstimator] = None,
):
    """
    Make the default AER executor collection.

    Args:
        experimentalist_pipeline: an experimentalist Pipeline to be wrapped
        experiment_runner_callable: an experiment runner function to be wrapped
        theorist_estimator: a scikit learn-compatible estimator to be wrapped

    Returns: A dictionary with keys "experimentalist", "experiment_runner", "theorist" and
        "full_cycle", with values which are Callables.


    Examples:

        If we make the empty executor collection, all the executors are no-ops:
        >>> make_default_online_executor_collection()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        mappingproxy({'experimentalist': <function no_op at 0x...>,
                      'experiment_runner': <function no_op at 0x...>,
                      'theorist': <function no_op at 0x...>,
                      'full_cycle': functools.partial(<function full_cycle_wrapper at 0x...>,
                                                      experimentalist_pipeline=None,
                                                      experiment_runner_callable=None,
                                                      theorist_estimator=None)})

        >>> from autora.experimentalist.pipeline import Pipeline
        >>> from sklearn.linear_model import LinearRegression
        >>> experimentalist_pipeline_ = Pipeline([('p', (1, 2))])
        >>> def experiment_runner_(x):
        ...     return 2 * x + 1
        >>> theorist_estimator_ = LinearRegression()
        >>> c = make_default_online_executor_collection(
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

    c = make_online_executor_collection(
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
