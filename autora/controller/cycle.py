"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from sklearn.base import BaseEstimator

from autora.controller.base import BaseController
from autora.controller.executor import make_default_online_executor_collection
from autora.controller.planner import full_cycle_planner
from autora.controller.state import Snapshot
from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


class Cycle(BaseController):
    """
    Runs an experimentalist, experiment runner, and theorist in a loop.

    Once initialized, the `cycle` can be started by calling `next(cycle)` or using the
        `cycle.run` method. Each iteration runs the full AER cycle, starting with the
        experimentalist and ending with the theorist.

    """

    def __init__(
        self,
        metadata: VariableCollection,
        theorist: Optional[BaseEstimator] = None,
        experimentalist: Optional[Pipeline] = None,
        experiment_runner: Optional[Callable] = None,
        params: Optional[Dict] = None,
        monitor: Optional[Callable[[Snapshot], None]] = None,
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
        """
        if params is None:
            params = {}
        state = Snapshot(
            metadata=metadata,
            conditions=[],
            observations=[],
            theories=[],
            params=params,
        )
        planner = full_cycle_planner

        self._experimentalist_pipeline = experimentalist
        self._experiment_runner_callable = experiment_runner
        self._theorist_estimator = theorist

        executor_collection = make_default_online_executor_collection(
            experimentalist_pipeline=self._experimentalist_pipeline,
            experiment_runner_callable=self._experiment_runner_callable,
            theorist_estimator=self._theorist_estimator,
        )

        super().__init__(
            state=state,
            planner=planner,
            executor_collection=executor_collection,
            monitor=monitor,
        )

    def run(self, num_cycles: int = 1):
        """Execute the next step in the cycle."""
        super().run(num_steps=num_cycles)
        return self

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
            >>> from autora.controller.cycle import Cycle
            >>> p = {"some": "params"}
            >>> c = Cycle(metadata=None, theorist=None, experimentalist=None,
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
            >>> from autora.controller.cycle import Cycle
            >>> from sklearn.linear_model import LinearRegression, PoissonRegressor
            >>> c = Cycle(metadata=None, theorist=LinearRegression(), experimentalist=None,
            ...                 experiment_runner=None)
            >>> c.theorist
            LinearRegression()

            >>> c.theorist = PoissonRegressor()
            >>> c.theorist
            PoissonRegressor()

        """
        return self._theorist_estimator

    @theorist.setter
    def theorist(self, value):
        self._theorist_estimator = value
        self.executor_collection = self._updated_executor_collection()

    @property
    def experimentalist(self):
        """
        Generates new experimental conditions.

        Examples:
            >>> from autora.controller.cycle import Cycle
            >>> from autora.experimentalist.pipeline import Pipeline
            >>> c = Cycle(metadata=None, theorist=None, experiment_runner=None,
            ...                 experimentalist=Pipeline([("pool", [11,12,13])]))
            >>> c.experimentalist
            Pipeline(steps=[('pool', [11, 12, 13])], params={})

            >>> c.experimentalist = Pipeline([('pool', [21,22,23])])
            >>> c.experimentalist
            Pipeline(steps=[('pool', [21, 22, 23])], params={})

        """
        return self._experimentalist_pipeline

    @experimentalist.setter
    def experimentalist(self, value):
        self._experimentalist_pipeline = value
        self.executor_collection = self._updated_executor_collection()

    @property
    def experiment_runner(self):
        """
        Generates new observations.

        Examples:
            >>> from autora.controller.cycle import Cycle
            >>> def plus_one(x): return x + 1
            >>> c = Cycle(metadata=None, theorist=None, experimentalist=None,
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
        return self._experiment_runner_callable

    @experiment_runner.setter
    def experiment_runner(self, value):
        self._experiment_runner_callable = value
        self.executor_collection = self._updated_executor_collection()

    def _updated_executor_collection(self):
        executor_collection = make_default_online_executor_collection(
            experimentalist_pipeline=self._experimentalist_pipeline,
            experiment_runner_callable=self._experiment_runner_callable,
            theorist_estimator=self._theorist_estimator,
        )
        return executor_collection
