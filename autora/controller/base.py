"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Generic, Mapping, Optional

from autora.controller.protocol import State

_logger = logging.getLogger(__name__)


class BaseController(Generic[State]):
    """
    Runs an experimentalist, theorist and experiment runner in a loop.

    Once initialized, the `controller` can be started by calling `next(controller)` or using the
        `controller.run` method.

    Attributes:
        state (CycleState or CycleStateHistory): an object which is updated during the cycle and
            is compatible with the `executor_collection`, `planner` and `monitor`.

        planner: a function which takes the `state` as input and returns the name one of the
            `executor_collection` names.

        executor_collection: a mapping between names and functions which take the state as
            input and return a state.

        monitor (Callable): a function which takes the state as input and is called at
            the end of each step.

    """

    def __init__(
        self,
        state: State,
        planner: Callable[[State], str],
        executor_collection: Mapping[str, Callable[[State], State]],
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
        next_function_name = self.planner(self.state)

        # Map
        next_function = self.executor_collection[next_function_name]
        next_params = self.state.params.get(next_function_name, {})

        # Execute
        result = next_function(self.state, params=next_params)

        # Update
        self.state = result

        # Monitor
        if self.monitor is not None:
            self.monitor(self.state)

        return self

    def __iter__(self):
        return self
