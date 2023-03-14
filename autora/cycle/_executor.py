from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable, Iterable, Protocol

import numpy as np

from autora.cycle.params import _resolve_state_params
from autora.cycle.state import SimpleCycleData
from autora.experimentalist.pipeline import Pipeline


class SupportsFit(Protocol):
    def fit(self, x, y, **params):
        ...


class OnlineExecutor:
    def __init__(
        self,
        experimentalist_pipeline: Pipeline,
        experiment_runner_callable: Callable,
        theorist_estimator: SupportsFit,
    ):
        self._experimentalist = experimentalist_pipeline
        self._experiment_runner = experiment_runner_callable
        self._theorist = theorist_estimator

    def experimentalist(self, state: SimpleCycleData, params: dict):
        new_conditions = self._experimentalist(**params)
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

        new_state = replace(
            state,
            conditions=state.conditions + [new_conditions_array],
        )

        return new_state

    def experiment_runner(self, state: SimpleCycleData, params: dict):
        x = state.conditions[-1]
        y = self._experiment_runner(x, **params)
        new_observations = np.column_stack([x, y])
        new_state = replace(state, observations=state.observations + [new_observations])
        return new_state

    def theorist(self, state: SimpleCycleData, params: dict):
        all_observations = np.row_stack(state.observations)
        n_xs = len(state.metadata.independent_variables)
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        if y.shape[1] == 1:
            y = y.ravel()
        new_theorist = copy.deepcopy(self._theorist)
        new_theorist.fit(x, y, **params)

        new_state = replace(
            state,
            theories=state.theories + [new_theorist],
        )

        return new_state


class FullCycleExecutor(OnlineExecutor):
    def full_cycle(self, state: SimpleCycleData):
        experimentalist_params = _resolve_state_params(state).get(
            "experimentalist", dict()
        )
        state = self.experimentalist(state, experimentalist_params)

        experiment_runner_params = _resolve_state_params(state).get(
            "experiment_runner", dict()
        )
        state = self.experiment_runner(state, experiment_runner_params)

        theorist_params = _resolve_state_params(state).get("theorist", dict())
        state = self.theorist(state, theorist_params)

        return state
