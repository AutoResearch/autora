"""
Objects for handling input and outputs from experimentalists, experiment runners and theorists.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable

import numpy as np
from sklearn.base import BaseEstimator

from autora.cycle.protocol.v1 import SupportsCycleState
from autora.cycle.state import resolve_state_params
from autora.experimentalist.pipeline import Pipeline


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

    def experimentalist(self, state: SupportsCycleState) -> SupportsCycleState:
        """Interface for running the experimentalist pipeline."""
        params = resolve_state_params(state).get("experimentalist", dict())
        new_conditions = self.experimentalist_pipeline(**params)

        assert isinstance(new_conditions, Iterable)
        # If the pipeline gives us an iterable, we need to make it into a concrete array.
        # We can't move this logic to the Pipeline, because the pipeline doesn't know whether
        # it's within another pipeline and whether it should convert the iterable to a
        # concrete array.
        new_conditions_values = list(new_conditions)
        new_conditions_array = np.array(new_conditions_values)

        assert isinstance(
            new_conditions_array, np.ndarray
        )  # Check the object is bounded
        new_state = state.update(conditions=[new_conditions_array])
        return new_state

    def experiment_runner(self, state: SupportsCycleState) -> SupportsCycleState:
        """Interface for running the experiment runner callable"""
        params = resolve_state_params(state).get("experiment_runner", dict())
        x = state.conditions[-1]
        y = self.experiment_runner_callable(x, **params)
        new_observations = np.column_stack([x, y])
        new_state = state.update(observations=[new_observations])
        return new_state

    def theorist(self, state: SupportsCycleState) -> SupportsCycleState:
        """Interface for running the theorist estimator."""
        params = resolve_state_params(state).get("theorist", dict())
        metadata = state.metadata
        observations = state.observations
        all_observations = np.row_stack(observations)
        n_xs = len(metadata.independent_variables)
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        if y.shape[1] == 1:
            y = y.ravel()
        new_theorist = copy.deepcopy(self.theorist_estimator)
        new_theorist.fit(x, y, **params)
        new_state = state.update(theories=[new_theorist])
        return new_state


class FullCycleExecutorCollection(OnlineExecutorCollection):
    """
    Runs a full AER cycle each `full_cycle` call in a single session.
    """

    def full_cycle(self, state: SupportsCycleState) -> SupportsCycleState:
        """
        Executes the experimentalist, experiment runner and theorist on the given state.

        Returns: A list of new results
        """
        experimentalist_result = self.experimentalist(state)
        experiment_runner_result = self.experiment_runner(experimentalist_result)
        theorist_result = self.theorist(experiment_runner_result)
        return theorist_result
