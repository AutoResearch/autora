""" Executors for the AutoRA Cycle."""

from __future__ import annotations

import copy
from typing import Callable, Iterable, List, Protocol

import numpy as np
from sklearn.base import BaseEstimator

from autora.cycle._state import (
    Result,
    ResultKind,
    SupportsData,
    _resolve_state_params,
    sequence_to_namespace,
)
from autora.experimentalist.pipeline import Pipeline


class Executor(Protocol):
    def __call__(self, state: SupportsData) -> Result:
        ...


class SupportsFullCycle(Protocol):
    full_cycle: Executor


class SupportsExperimentalistExperimentRunnerTheorist(Protocol):
    experimentalist: Executor
    experiment_runner: Executor
    theorist: Executor


class OnlineExecutorCollection:
    """
    Runs experiment design, observation and theory generation in a single session.

    This object allows a user to specify
    - an experimentalist: a Pipeline
    - an experiment runner: some Callable and
    - a theorist: a scikit-learn-compatible estimator with a fit method

    ... and exposes methods to call these and update a CycleState object with new data.
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

    def experimentalist(self, state: List[Result]) -> List[Result]:
        """Interface for running the experimentalist pipeline."""
        params = _resolve_state_params(state).get("experimentalist", dict())
        new_conditions = self.experimentalist_pipeline(**params)
        if isinstance(new_conditions, Iterable):
            # If the pipeline gives us an iterable, we need to make it into a concrete array.
            # We can't move this logic to the Pipeline, because the pipeline doesn't know whether
            # it's within another pipeline and whether it should convert the iterable to a
            # concrete array.
            new_conditions_values = list(new_conditions)
            new_conditions_array = np.array(new_conditions_values)
        else:
            raise NotImplementedError(f"Object {new_conditions} can't be handled yet.")
        assert isinstance(
            new_conditions_array, np.ndarray
        )  # Check the object is bounded
        result = [Result(new_conditions_array, kind=ResultKind.CONDITION)]
        return result

    def experiment_runner(self, state: List[Result]) -> List[Result]:
        """Interface for running the experiment runner callable"""
        params = _resolve_state_params(state).get("experiment_runner", dict())
        x = sequence_to_namespace(state).conditions[-1]
        y = self.experiment_runner_callable(x, **params)
        new_observations = np.column_stack([x, y])
        result = [Result(new_observations, kind=ResultKind.OBSERVATION)]
        return result

    def theorist(self, state: List[Result]) -> List[Result]:
        """Interface for running the theorist estimator."""
        params = _resolve_state_params(state).get("theorist", dict())
        metadata = sequence_to_namespace(state).metadata
        observations = sequence_to_namespace(state).observations
        all_observations = np.row_stack(observations)
        n_xs = len(metadata.independent_variables)
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        if y.shape[1] == 1:
            y = y.ravel()
        new_theorist = copy.deepcopy(self.theorist_estimator)
        new_theorist.fit(x, y, **params)
        result = [Result(new_theorist, kind=ResultKind.THEORY)]
        return result


class FullCycleExecutorCollection(OnlineExecutorCollection):
    """
    Runs a full AER cycle each `full_cycle` call in a single session.
    """

    def full_cycle(self, state: List[Result]) -> List[Result]:
        state_ = list(state)
        experimentalist_result = self.experimentalist(state_)
        experiment_runner_result = self.experiment_runner(
            state_ + experimentalist_result
        )
        theorist_result = self.theorist(
            state_ + experimentalist_result + experiment_runner_result
        )
        return experimentalist_result + experiment_runner_result + theorist_result
