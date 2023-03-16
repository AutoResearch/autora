"""
The basic cycle controller for AER.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Callable, Dict, Optional

from autora.cycle.executor import FullCycleExecutorCollection
from autora.cycle.planner import full_cycle_planner
from autora.cycle.state import history_to_kind, init_result_list
from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


class SimpleCycle:
    """
    Runs an experimentalist, theorist and experiment runner in a loop.

    Once initialized, the `cycle` can be started using the `cycle.run` method
        or by calling `next(cycle)`.

    The `.data` attribute is updated with the results.

    Attributes:
        data (dataclass): an object which is updated during the cycle and has the following
            properties:

            - `metadata`
            -  `params` (dict): a nested dictionary with parameters for the cycle parts.
                    `{
                        "experimentalist": {<experimentalist params...>},
                        "theorist": {<theorist params...>},
                        "experiment_runner": {<experiment_runner params...>}
                    }`
            - `conditions`: a list of np.ndarrays representing all of the IVs proposed by the
                experimentalist
            - `observations`: a list of np.ndarrays representing all of the IVs and DVs returned by
                the experiment runner
            - `theories`: a list of all the fitted theories (scikit-learn compatible estimators)

        executor_collection: an object with interfaces for running the theorist, experimentalist
            and experiment_runner.

    Examples:

        ### Basic Usage

        Aim: Use the SimpleCycle to recover a simple ground truth theory from noisy data.

        >>> def ground_truth(x):
        ...     return x + 1

        The space of allowed x values is the integers between 0 and 10 inclusive,
        and we record the allowed output values as well.
        >>> from autora.variable import VariableCollection, Variable
        >>> metadata_0 = VariableCollection(
        ...    independent_variables=[Variable(name="x1", allowed_values=range(11))],
        ...    dependent_variables=[Variable(name="y", value_range=(-20, 20))],
        ...    )

        The experimentalist is used to propose experiments.
        Since the space of values is so restricted, we can just sample them all each time.
        >>> from autora.experimentalist.pipeline import make_pipeline
        >>> example_experimentalist = make_pipeline(
        ...     [metadata_0.independent_variables[0].allowed_values])

        When we run a synthetic experiment, we get a reproducible noisy result:
        >>> import numpy as np
        >>> def get_example_synthetic_experiment_runner():
        ...     rng = np.random.default_rng(seed=180)
        ...     def runner(x):
        ...         return ground_truth(x) + rng.normal(0, 0.1, x.shape)
        ...     return runner
        >>> example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()
        >>> example_synthetic_experiment_runner(np.array([1]))
        array([2.04339546])

        The theorist "tries" to work out the best theory.
        We use a trivial scikit-learn regressor.
        >>> from sklearn.linear_model import LinearRegression
        >>> example_theorist = LinearRegression()

        We initialize the SimpleCycle with the metadata describing the domain of the theory,
        the theorist, experimentalist and experiment runner,
        as well as a monitor which will let us know which cycle we're currently on.
        >>> cycle = SimpleCycle(
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ...     monitor=lambda cycle: print(f"Generated {len(cycle.data.theories)} theories"),
        ... )
        >>> cycle # doctest: +ELLIPSIS
        <...SimpleCycle object at 0x...>

        We can run the cycle by calling the run method:
        >>> cycle.run(num_cycles=3)  # doctest: +ELLIPSIS
        Generated 1 theories
        Generated 2 theories
        Generated 3 theories
        <...SimpleCycle object at 0x...>

        We can now interrogate the results. The first set of conditions which went into the
        experiment runner were:
        >>> cycle.data.conditions[0]
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        The observations include the conditions and the results:
        >>> cycle.data.observations[0]
        array([[ 0.        ,  0.92675345],
               [ 1.        ,  1.89519928],
               [ 2.        ,  3.08746571],
               [ 3.        ,  3.93023943],
               [ 4.        ,  4.95429102],
               [ 5.        ,  6.04763988],
               [ 6.        ,  7.20770574],
               [ 7.        ,  7.85681519],
               [ 8.        ,  9.05735823],
               [ 9.        , 10.18713406],
               [10.        , 10.88517906]])

        In the third cycle (index = 2) the first and last values are different again:
        >>> cycle.data.observations[2][[0,-1]]
        array([[ 0.        ,  1.08559827],
               [10.        , 11.08179553]])

        The best fit theory after the first cycle is:
        >>> cycle.data.theories[0]
        LinearRegression()

        >>> def report_linear_fit(m: LinearRegression,  precision=4):
        ...     s = f"y = {np.round(m.coef_[0].item(), precision)} x " \\
        ...     f"+ {np.round(m.intercept_.item(), 4)}"
        ...     return s
        >>> report_linear_fit(cycle.data.theories[0])
        'y = 1.0089 x + 0.9589'

        The best fit theory after all the cycles, including all the data, is:
        >>> report_linear_fit(cycle.data.theories[-1])
        'y = 0.9989 x + 1.0292'

        This is close to the ground truth theory of x -> (x + 1)

        We can also run the cycle with more control over the execution flow:
        >>> next(cycle) # doctest: +ELLIPSIS
        Generated 4 theories
        <...SimpleCycle object at 0x...>

        >>> next(cycle) # doctest: +ELLIPSIS
        Generated 5 theories
        <...SimpleCycle object at 0x...>

        >>> next(cycle) # doctest: +ELLIPSIS
        Generated 6 theories
        <...SimpleCycle object at 0x...>

        We can continue to run the cycle as long as we like,
        with a simple arbitrary stopping condition like the number of theories generated:
        >>> from itertools import takewhile
        >>> _ = list(takewhile(lambda c: len(c.data.theories) < 9, cycle))
        Generated 7 theories
        Generated 8 theories
        Generated 9 theories

        ... or the precision (here we keep iterating while the difference between the gradients
        of the second-last and last cycle is larger than 1x10^-3).
        >>> _ = list(
        ...         takewhile(
        ...             lambda c: np.abs(c.data.theories[-1].coef_.item() -
        ...                            c.data.theories[-2].coef_.item()) > 1e-3,
        ...             cycle
        ...         )
        ...     )
        Generated 10 theories
        Generated 11 theories

        ... or continue to run as long as we like:
        >>> _ = cycle.run(num_cycles=100) # doctest: +ELLIPSIS
        Generated 12 theories
        ...
        Generated 111 theories

        ### Passing Static Parameters

        It's easy to pass parameters to the cycle components, if there are any needed.
        Here we have an experimentalist which takes a parameter:
        >>> uniform_random_rng = np.random.default_rng(180)
        >>> def uniform_random_sampler(n):
        ...     return uniform_random_rng.uniform(low=0, high=11, size=n)
        >>> example_experimentalist_with_parameters = make_pipeline([uniform_random_sampler])

        The cycle can handle that using the `params` keyword:
        >>> cycle_with_parameters = SimpleCycle(
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist_with_parameters,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ...     params={"experimentalist": {"uniform_random_sampler": {"n": 7}}}
        ... )
        >>> _ = cycle_with_parameters.run()
        >>> cycle_with_parameters.data.conditions[-1].flatten()
        array([6.33661987, 7.34916618, 6.08596494, 2.28566582, 1.9553974 ,
               5.80023149, 3.27007909])

        For the next cycle, if we wish, we can change the parameter value:
        >>> cycle_with_parameters.params["experimentalist"]["uniform_random_sampler"]\\
        ...     ["n"] = 2
        >>> _ = cycle_with_parameters.run()
        >>> cycle_with_parameters.data.conditions[-1].flatten()
        array([10.5838232 ,  9.45666031])

        ### Accessing "State-dependent Properties"

         Some experimentalists, experiment runners and theorists require access to the values
            created during the cycle execution, e.g. experimentalists which require access
            to the current best theory or the observed data. These data update each cycle, and
            so cannot easily be set using simple `params`.

        For this case, it is possible to use "state-dependent properties" in the `params`
            dictionary. These are the following strings, which will be replaced during execution by
            their respective current values:

        - `"%observations.ivs[-1]%"`: the last observed independent variables
        - `"%observations.dvs[-1]%"`: the last observed dependent variables
        - `"%observations.ivs%"`: all the observed independent variables,
            concatenated into a single array
        - `"%observations.dvs%"`: all the observed dependent variables,
            concatenated into a single array
        - `"%theories[-1]%"`: the last fitted theorist
        - `"%theories%"`: all the fitted theorists

        In the following example, we use the `"observations.ivs"` cycle property for an
            experimentalist which excludes those conditions which have
            already been seen.

        >>> metadata_1 = VariableCollection(
        ...    independent_variables=[Variable(name="x1", allowed_values=range(10))],
        ...    dependent_variables=[Variable(name="y")],
        ...    )
        >>> random_sampler_rng = np.random.default_rng(seed=180)
        >>> def custom_random_sampler(conditions, n):
        ...     sampled_conditions = random_sampler_rng.choice(conditions, size=n, replace=False)
        ...     return sampled_conditions
        >>> def exclude_conditions(conditions, excluded_conditions):
        ...     remaining_conditions = list(set(conditions) - set(excluded_conditions.flatten()))
        ...     return remaining_conditions
        >>> unobserved_data_experimentalist = make_pipeline([
        ...     metadata_1.independent_variables[0].allowed_values,
        ...     exclude_conditions,
        ...     custom_random_sampler
        ...     ]
        ... )
        >>> cycle_with_state_dep_properties = SimpleCycle(
        ...     metadata=metadata_1,
        ...     theorist=example_theorist,
        ...     experimentalist=unobserved_data_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ...     params={
        ...         "experimentalist": {
        ...             "exclude_conditions": {"excluded_conditions": "%observations.ivs%"},
        ...             "custom_random_sampler": {"n": 1}
        ...         }
        ...     }
        ... )

        Now we can run the cycler to generate conditions and run experiments. The first time round,
        we have the full set of 10 possible conditions to select from, and we select "2" at random:
        >>> _ = cycle_with_state_dep_properties.run()
        >>> cycle_with_state_dep_properties.data.conditions[-1]
        array([2])

        We can continue to run the cycler, each time we add more to the list of "excluded" options:
        >>> _ = cycle_with_state_dep_properties.run(num_cycles=5)
        >>> cycle_with_state_dep_properties.data.conditions
        [array([2]), array([6]), array([5]), array([7]), array([3]), array([4])]

        By using the monitor callback, we can investigate what's going on with the
        state-dependent properties:
        >>> cycle_with_state_dep_properties.monitor = lambda cycle: print(
        ...     np.row_stack(cycle.data.observations)[:,0]  # just the independent variable values
        ... )

        The monitor evaluates at the end of each cycle
        and shows that we've added a new observed IV each step
        >>> _ = cycle_with_state_dep_properties.run()
        [2. 6. 5. 7. 3. 4. 9.]
        >>> _ = cycle_with_state_dep_properties.run()
        [2. 6. 5. 7. 3. 4. 9. 0.]

        We deactivate the monitor by making it "None" again.
        >>> cycle_with_state_dep_properties.monitor = None

        We can continue until we've sampled all of the options:
        >>> _ = cycle_with_state_dep_properties.run(num_cycles=2)
        >>> cycle_with_state_dep_properties.data.conditions # doctest: +NORMALIZE_WHITESPACE
        [array([2]), array([6]), array([5]), array([7]), array([3]), \
        array([4]), array([9]), array([0]), array([8]), array([1])]

        If we try to evaluate it again, the experimentalist fails, as there aren't any more
        conditions which are available:
        >>> cycle_with_state_dep_properties.run()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: a cannot be empty unless no samples are taken

        ### Using Alternative Executors and Planners

        By switching out the `executor_collection` and/or the `planner`, we can specify a
        different way of running the cycle.

        #### Easier Seeding with a Smarter Planner

        In this example, we use a planner which considers the last available result and picks the
        matching next step. This means that seeding is relatively simple.
        >>> from autora.cycle.executor import OnlineExecutorCollection
        >>> from autora.cycle.planner import last_result_kind_planner
        >>> def monitor(cycle):
        ...     print(f"MONITOR: Generated new {cycle.history[-1].kind}")
        >>> cycle_with_last_result_planner = SimpleCycle(
        ...     planner=last_result_kind_planner,
        ...     executor_collection=OnlineExecutorCollection,
        ...     monitor=monitor,
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ... )

        When we run this cycle starting with no data, we generate an experimental condition first:
        >>> _ = list(takewhile(lambda c: len(c.data.theories) < 2, cycle_with_last_result_planner))
        MONITOR: Generated new CONDITION
        MONITOR: Generated new OBSERVATION
        MONITOR: Generated new THEORY
        MONITOR: Generated new CONDITION
        MONITOR: Generated new OBSERVATION
        MONITOR: Generated new THEORY

        However, if we seed the same cycle with observations, then its first Executor will be the
        theorist:
        >>> from autora.cycle.state import Result
        >>> cycle_with_seed_observation = SimpleCycle(
        ...     planner=last_result_kind_planner,
        ...     executor_collection=OnlineExecutorCollection,
        ...     monitor=monitor,
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ... )
        >>> seed_observation = example_synthetic_experiment_runner(np.linspace(0,5,10))
        >>> cycle_with_seed_observation.history.append(Result(seed_observation,kind="OBSERVATION"))

        >>> _ = next(cycle_with_seed_observation)
        MONITOR: Generated new THEORY

        #### Arbitrary Execution Order (Toy Example)

        In the next example, we use a planner which suggests a different random operation at each
        step, demonstrating arbitrary execution order.

        This might be useful in cases when different experimentalists or theorists are needed at
        different times in the cycle, e.g. for initial seeding.
        >>> from autora.cycle.planner import random_operation_planner
        >>> def monitor(cycle):
        ...     print(f"MONITOR: Generated new {cycle.history[-1].kind}")
        >>> cycle_with_random_planner = SimpleCycle(
        ...     planner=random_operation_planner,
        ...     executor_collection=OnlineExecutorCollection,
        ...     monitor=monitor,
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ... )

        The `random_operation_planner` depends on the python random number generator, so we seed
        it first:
        >>> from random import seed
        >>> seed(42)

        We also want to watch the logging messages from the cycle:
        >>> import logging
        >>> import sys
        >>> logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout,
        ...     level=logging.INFO)

        Now we can evaluate the cycle and watch its behaviour:
        >>> for i in range(9):
        ...     print(f"==== Starting iteration {i=} ====")
        ...     try:
        ...         _ = next(cycle_with_random_planner)
        ...     except ValueError as e:
        ...         print(f"FAILED: on iteration {i=} time with {e=}") # doctest: +ELLIPSIS
        ==== Starting iteration i=0 ====
        FAILED: on iteration i=0 time with e=ValueError('need at least one array to concatenate')
        ==== Starting iteration i=1 ====
        MONITOR: Generated new CONDITION
        ==== Starting iteration i=2 ====
        MONITOR: Generated new CONDITION
        ...
        ==== Starting iteration i=8 ====
        MONITOR: Generated new THEORY

        The first cycle, the theorist is selected as the random Executor, and it fails because it
        depends on there being observations to theorize against.

        By the last iteration, there are observations which the theorist can use, and it succeeds.

    """

    def __init__(
        self,
        metadata: VariableCollection,
        theorist,
        experimentalist,
        experiment_runner,
        monitor: Optional[Callable[[SimpleCycle], None]] = None,
        params: Optional[Dict] = None,
        executor_collection=FullCycleExecutorCollection,
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
            executor_collection: an object with functions for running the theorist,
                experimentalist and experiment_runner.
        """

        self.executor_collection = executor_collection(
            theorist_estimator=theorist,
            experiment_runner_callable=experiment_runner,
            experimentalist_pipeline=experimentalist,
        )

        self.monitor = monitor
        if params is None:
            params = dict()

        self.history = init_result_list(
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
        next_function = self.planner(self.history, self.executor_collection)

        # Execute
        result = next_function(self.history)

        # Update
        self.history = self.history + result

        # Monitor
        self._monitor_callback()

        return self

    def __iter__(self):
        return self

    def _monitor_callback(self):
        if self.monitor is not None:
            self.monitor(self)

    @property
    def data(self) -> SimpleNamespace:
        """
        View the cycle's data as a namespace with attributes:
        - `.metadata`
        - `.params`
        - `.conditions`
        - `.observations`
        - `.theories`

        """
        return history_to_kind(self.history)

    @property
    def params(self):
        """
        Examples:
            >>> from autora.cycle.simple import SimpleCycle
            >>> p = {"some": "params"}
            >>> c = SimpleCycle(metadata=None, theorist=None, experimentalist=None,
            ...                 experiment_runner=None, params=p)
            >>> c.params
            {'some': 'params'}

            >>> c.params = {"new": "value"}
            >>> c.params
            {'new': 'value'}
        """
        return self.data.params

    @params.setter
    def params(self, value):
        self.history = self.history + init_result_list(params=value)

    @property
    def theorist(self):
        """
        Examples:
            >>> from autora.cycle.simple import SimpleCycle
            >>> from sklearn.linear_model import LinearRegression, PoissonRegressor
            >>> c = SimpleCycle(metadata=None, theorist=LinearRegression(), experimentalist=None,
            ...                 experiment_runner=None)
            >>> c.theorist
            LinearRegression()

            >>> c.theorist = PoissonRegressor()
            >>> c.theorist
            PoissonRegressor()

        """
        """The theorist – scikit-learn-compatible estimator – used when running the cycle."""
        return self.executor_collection.theorist_estimator

    @theorist.setter
    def theorist(self, value):
        self.executor_collection.theorist_estimator = value

    @property
    def experimentalist(self):
        """
        The experimentalist Pipeline used when running the cycle.

        Examples:
            >>> from autora.cycle.simple import SimpleCycle
            >>> from autora.experimentalist.pipeline import Pipeline

            z
            >>> c = SimpleCycle(metadata=None, theorist=None, experiment_runner=None,
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
        """The callable used to generate new experimental results"""
        return self.executor_collection.experiment_runner_callable

    @experiment_runner.setter
    def experiment_runner(self, value):
        self.executor_collection.experiment_runner_callable = value
