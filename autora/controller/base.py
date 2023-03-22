"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Mapping, Optional, TypeVar, Union

from autora.controller.protocol import (
    SupportsControllerState,
    SupportsControllerStateHistory,
)

_logger = logging.getLogger(__name__)


State = TypeVar(
    "State", bound=Union[SupportsControllerState, SupportsControllerStateHistory]
)
ExecutorName = TypeVar("ExecutorName", bound=str)


class BaseController:
    """
    Runs an experimentalist, theorist and experiment runner in a loop.

    Once initialized, the `controller` can be started by calling `next(controller)` or using the
        `controller.run` method.

    Attributes:
        state (CycleState or CycleStateHistory): an object which is updated during the cycle and
            has the following properties:

            - `metadata` (VariableCollection)
            -  `params` (dict): a nested dictionary with parameters for the cycle parts.
                    `{
                        "experimentalist": {<experimentalist params...>},
                        "theorist": {<theorist params...>},
                        "experiment_runner": {<experiment_runner params...>}
                    }`
            - `conditions`: a list of ArrayLike objects representing all the IVs proposed by the
                experimentalist
            - `observations`: a list of ArrayLike objects representing all the IVs and DVs
                returned by the experiment runner
            - `theories`: a list of all the fitted theories (scikit-learn compatible estimators)
            - `history`: (only when using CycleStateHistory) a sequential list of all the above.

        executor_collection (FullCycleExecutorCollection, OnlineExecutorCollection): an
            object with interfaces for running the theorist, experimentalist and
            experiment_runner. This must be compatible with the `state`.

        planner (Callable): a function which takes the `state` as input and returns one of the
            `executor_collection` methods. This must be compatible with both the `state` and
            the `executor_collection`.

        monitor (Callable): a function which takes the controller as input and is called at
            the end of each step.

    """

    def __init__(
        self,
        state: State,
        planner: Callable[[State], ExecutorName],
        executor_collection: Mapping[ExecutorName, Callable[[State], State]],
        monitor: Optional[Callable[[State], None]] = None,
    ):
        """
        Args:
            state: a fully instantiated controller state object compatible with the planner,
                executor_collection and monitor
            planner: a function which maps from the state to the next ExecutorName
            executor_collection: a mapping from the ExecutorName to a callable which can operate
                on the state and return an updated state
            monitor: a function which takes the state object as input
        """

        self.state = state
        self.planner = planner
        self.executor_collection = executor_collection
        self.monitor = monitor

    def run(self, num_steps: int = 1):
        """Execute the next step in the cycle."""
        for i in range(num_steps):
            next(self)
        return self

    def __next__(self):

        # Plan
        next_function_name = self.planner(self.state, self.executor_collection)

        # Map
        next_function = self.executor_collection[next_function_name]

        # Execute
        result = next_function(self.state)

        # Update
        self.state = result

        # Monitor
        if self.monitor is not None:
            self.monitor(self)

        return self

    def __iter__(self):
        return self
