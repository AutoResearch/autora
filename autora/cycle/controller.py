"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from autora.cycle.executor import FullCycleExecutorCollection
from autora.cycle.planner import full_cycle_planner
from autora.cycle.state import CycleStateHistory
from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


class Controller:
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
        metadata: VariableCollection,
        theorist,
        experimentalist,
        experiment_runner,
        monitor: Optional[Callable[[Controller], None]] = None,
        params: Optional[Dict] = None,
        executor_collection=FullCycleExecutorCollection,
        state_collection=CycleStateHistory,
        planner=full_cycle_planner,
    ):
        """
        Args:
            metadata: a description of the dependent and independent variables
            theorist: a scikit-learn-compatible estimator
            experimentalist: an autora.experimentalist.Pipeline
            experiment_runner: a function to map independent variables onto observed dependent
                variables
            monitor: a function which gets read-only access to the `data` attribute at the end of
                each cycle.
            params: a nested dictionary with parameters to be passed to the parts of the cycle.
                E.g. if the experimentalist had a step named "pool" which took an argument "n",
                which you wanted to set to the value 30, then params would be set to this:
                `{"experimentalist": {"pool": {"n": 30}}}`
            executor_collection: the constructor for an object with functions for running the
                theorist, experimentalist and experiment_runner.
            state_collection: the constructor for an object which holds the state of the cycle
                and is passed between executors
        """

        self.executor_collection = executor_collection(
            theorist_estimator=theorist,
            experiment_runner_callable=experiment_runner,
            experimentalist_pipeline=experimentalist,
        )

        self.monitor = monitor
        if params is None:
            params = dict()

        self.state = state_collection(
            metadata=metadata,
            conditions=[],
            observations=[],
            theories=[],
            params=params,
        )

        self.planner = planner

    def run(self, num_cycles: int = 1):
        """Execute the next step in the cycle."""
        for i in range(num_cycles):
            next(self)
        return self

    def __next__(self):

        # Plan
        next_function = self.planner(self.state, self.executor_collection)

        # Execute
        result = next_function(self.state)

        # Update
        self.state = result

        # Monitor
        self._monitor_callback()

        return self

    def __iter__(self):
        return self

    def _monitor_callback(self):
        if self.monitor is not None:
            self.monitor(self)

    @property
    def data(self):
        """An alias for `.state`."""
        return self.state

    @property
    def params(self):
        """
        The parameters passed to the `theorist`, `experimentalist` and `experiment_runner`.

        Should be a nested dictionary like
        ```
        {'experimentalist': {... params for experimentalist ...},
         'experiment_runner': {... params for experiment_runner ...},
         'theorist': {... params for theorist ...}}
        ```


        Examples:
            >>> from autora.cycle.controller import Controller
            >>> p = {"some": "params"}
            >>> c = Controller(metadata=None, theorist=None, experimentalist=None,
            ...                 experiment_runner=None, params=p)
            >>> c.params
            {'some': 'params'}

            >>> c.params = {"new": "value"}
            >>> c.params
            {'new': 'value'}
        """
        return self.state.params

    @params.setter
    def params(self, value):
        self.state = self.state.update(params=value)

    @property
    def theorist(self):
        """
        Generates new theories.

        Examples:
            >>> from autora.cycle.controller import Controller
            >>> from sklearn.linear_model import LinearRegression, PoissonRegressor
            >>> c = Controller(metadata=None, theorist=LinearRegression(), experimentalist=None,
            ...                 experiment_runner=None)
            >>> c.theorist
            LinearRegression()

            >>> c.theorist = PoissonRegressor()
            >>> c.theorist
            PoissonRegressor()

        """
        return self.executor_collection.theorist_estimator

    @theorist.setter
    def theorist(self, value):
        self.executor_collection.theorist_estimator = value

    @property
    def experimentalist(self):
        """
        Generates new experimental conditions.

        Examples:
            >>> from autora.cycle.controller import Controller
            >>> from autora.experimentalist.pipeline import Pipeline
            >>> c = Controller(metadata=None, theorist=None, experiment_runner=None,
            ...                 experimentalist=Pipeline([("pool", [11,12,13])]))
            >>> c.experimentalist
            Pipeline(steps=[('pool', [11, 12, 13])], params={})

            >>> c.experimentalist = Pipeline([('pool', [21,22,23])])
            >>> c.experimentalist
            Pipeline(steps=[('pool', [21, 22, 23])], params={})

        """
        return self.executor_collection.experimentalist_pipeline

    @experimentalist.setter
    def experimentalist(self, value):
        self.executor_collection.experimentalist_pipeline = value

    @property
    def experiment_runner(self):
        """
        Generates new observations.

        Examples:
            >>> from autora.cycle.controller import Controller
            >>> def plus_one(x): return x + 1
            >>> c = Controller(metadata=None, theorist=None, experimentalist=None,
            ...                 experiment_runner=plus_one)
            >>> c.experiment_runner  # doctest: +ELLIPSIS
            <function plus_one at 0x...>
            >>> c.experiment_runner(1)
            2

            >>> def plus_two(x): return x + 2
            >>> c.experiment_runner = plus_two
            >>> c.experiment_runner  # doctest: +ELLIPSIS
            <function plus_two at 0x...>
            >>> c.experiment_runner(1)
            3

        """
        """The callable used to generate new experimental results"""
        return self.executor_collection.experiment_runner_callable

    @experiment_runner.setter
    def experiment_runner(self, value):
        self.executor_collection.experiment_runner_callable = value
