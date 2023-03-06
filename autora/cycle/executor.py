import copy
from typing import Callable, Iterable

import numpy as np
from sklearn.base import BaseEstimator

from autora.cycle.protocol.v1 import Cycle
from autora.cycle.result import Result, ResultKind
from autora.experimentalist.pipeline import Pipeline


def wrap_theorist_scikit_learn(theorist: BaseEstimator):
    def wrapped_theorist(cycle: Cycle):
        params = cycle.params.get("theorist", dict())
        all_observations = np.row_stack(cycle.state.observations)
        n_xs = len(cycle.state.metadata.independent_variables)
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        if y.shape[1] == 1:
            y = y.ravel()
        new_theorist = copy.deepcopy(theorist)
        new_theorist.fit(x, y, **params)
        result = Result(data=new_theorist, kind=ResultKind.THEORY)
        return result

    return wrapped_theorist


def wrap_experimentalist_autora_experimentalist_pipeline(experimentalist: Pipeline):
    def wrapped_experimentalist(cycle: Cycle):

        params = cycle.params.get("experimentalist", dict())
        new_conditions = experimentalist(**params)
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
        result = Result(data=new_conditions_array, kind=ResultKind.CONDITION)
        return result

    return wrapped_experimentalist


def wrap_experiment_runner_synthetic_experiment(experiment_runner: Callable):
    def wrapped_experiment_runner(cycle):
        params = cycle.params.get("experiment_runner", dict())
        x = cycle.state.conditions[-1]
        y = experiment_runner(x, **params)
        new_observations = np.column_stack([x, y])
        result = Result(data=new_observations, kind=ResultKind.OBSERVATION)
        return result

    return wrapped_experiment_runner
