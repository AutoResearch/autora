""" Filesystem Cycle"""

import logging
from typing import Callable, Dict, Optional, Sequence, Union

from sklearn.base import BaseEstimator

from autora.cycle.executor import (
    wrap_experiment_runner_synthetic_experiment,
    wrap_experimentalist_autora_experimentalist_pipeline,
    wrap_theorist_scikit_learn,
)
from autora.cycle.planner import last_result_kind_planner
from autora.cycle.protocol.v1 import Planner
from autora.cycle.result import Result, ResultCollection
from autora.cycle.result.serializer import ResultCollectionSerializer
from autora.cycle.simple import _get_cycle_properties, _resolve_cycle_properties
from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


class FilesystemCycle:
    def __init__(
        self,
        theorist: BaseEstimator,
        experimentalist: Pipeline,
        experiment_runner: Callable,
        monitor: Optional[Callable[[ResultCollection], None]] = None,
        raw_params: Optional[Dict] = None,
        planner: Planner = last_result_kind_planner,
        # Load Parameters
        metadata: Optional[VariableCollection] = None,
        results: Optional[Union[Sequence[Result], Result]] = None,
        state: Optional[Union[ResultCollectionSerializer, ResultCollection]] = None,
        # Dump Parameters
        serializer: Optional[ResultCollectionSerializer] = None,
    ):
        """
        Args:
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
            metadata: a VariableCollection describing the domain of the problem.
                Incompatible with setting `state`
            result_collection: a ResultCollection, sequence of Results or single Result which
                will be used to seed the controller. Incompatible with setting `state`.
            state: a ResultCollection (or a ...Serializer which can load one) which includes
                both the Metadata and the Results.


        Examples:

            ### Basic Usage

            Aim: Use the FilesystemCycle to recover a simple ground truth theory from noisy data.

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
            >>> example_synthetic_experiment_runner(np.ndarray([1]))
            array([2.04339546])

            The theorist "tries" to work out the best theory.
            We use a trivial scikit-learn regressor.
            >>> from sklearn.linear_model import LinearRegression
            >>> example_theorist = LinearRegression()

            We initialize the FilesystemCycle with the metadata describing the domain of the theory,
            the theorist, experimentalist and experiment runner,
            as well as a monitor which will let us know which cycle we're currently on.
            >>> def last_datum_monitor(state):
            ...     print(f"Generated {len(state)}-th datum, a new {state[-1].kind}")
            >>> cycle = FilesystemCycle(
            ...     theorist=example_theorist,
            ...     experimentalist=example_experimentalist,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     monitor=last_datum_monitor,
            ...     metadata=metadata_0)
            >>> cycle # doctest: +ELLIPSIS
            <filesystem.FilesystemCycle object at 0x...>

            We can run the cycle by calling the run method:
            >>> cycle.run(num_steps=3)  # doctest: +ELLIPSIS
            Generated 1-th datum, a new CONDITION
            Generated 2-th datum, a new OBSERVATION
            Generated 3-th datum, a new THEORY
            <filesystem.FilesystemCycle object at 0x...>

            To run the full cycle three times, we need 9 steps in total:
            >>> cycle.run(num_steps=6)  # doctest: +ELLIPSIS
            Generated 4-th datum, a new CONDITION
            ...
            Generated 9-th datum, a new THEORY
            <filesystem.FilesystemCycle object at 0x...>

            We can now interrogate the results. The first set of conditions which went into the
            experiment runner were:
            >>> cycle.state.conditions[0]
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

            The observations include the conditions and the results:
            >>> cycle.state.observations[0]
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
            >>> cycle.state.observations[2][[0,-1]]
            array([[ 0.        ,  1.08559827],
                   [10.        , 11.08179553]])

            The best fit theory after the first cycle is:
            >>> cycle.state.theories[0]
            LinearRegression()

            >>> def report_linear_fit(m: LinearRegression,  precision=4):
            ...     s = f"y = {np.round(m.coef_[0].item(), precision)} x " \\
            ...     f"+ {np.round(m.intercept_.item(), 4)}"
            ...     return s
            >>> report_linear_fit(cycle.state.theories[0])
            'y = 1.0089 x + 0.9589'

            The best fit theory after all the cycles, including all the data, is:
            >>> report_linear_fit(cycle.state.theories[-1])
            'y = 0.9989 x + 1.0292'

            This is close to the ground truth theory of x -> (x + 1)

            We can also run the cycle with more control over the execution flow:
            >>> next(cycle) # doctest: +ELLIPSIS
            Generated 10-th datum, a new CONDITION
            <filesystem.FilesystemCycle object at 0x...>

            >>> next(cycle) # doctest: +ELLIPSIS
            Generated 11-th datum, a new OBSERVATION
            <filesystem.FilesystemCycle object at 0x...>

            >>> next(cycle) # doctest: +ELLIPSIS
            Generated 12-th datum, a new THEORY
            <filesystem.FilesystemCycle object at 0x...>

            We can continue to run the cycle as long as we like,
            with a simple arbitrary stopping condition like the number of theories generated:
            >>> from itertools import takewhile
            >>> _ = list(takewhile(lambda c: len(c.state.theories) < 9, cycle)) # doctest: +ELLIPSIS
            Generated 13-th datum, a new CONDITION
            ...
            Generated 27-th datum, a new THEORY

            ... or the precision (here we keep iterating while the difference between the gradients
            of the second-last and last cycle is larger than 1x10^-3).
            >>> _ = list(
            ...         takewhile(
            ...             lambda c: np.abs(c.state.theories[-1].coef_.item() -
            ...                            c.state.theories[-2].coef_.item()) > 1e-3,
            ...             cycle
            ...         )
            ...     ) # doctest: +ELLIPSIS
            Generated 28-th datum, a new CONDITION
            ...
            Generated 33-th datum, a new THEORY

            ... or continue to run as long as we like:
            >>> _ = cycle.run(num_steps=102) # doctest: +ELLIPSIS
            Generated 34-th datum, a new CONDITION
            ...
            Generated 135-th datum, a new THEORY

            ### Passing Static Parameters

            It's easy to pass parameters to the cycle components, if there are any needed.
            Here we have an experimentalist which takes a parameter:
            >>> uniform_random_rng = np.random.default_rng(180)
            >>> def uniform_random_sampler(n):
            ...     return uniform_random_rng.uniform(low=0, high=11, size=n)
            >>> example_experimentalist_with_parameters = make_pipeline([uniform_random_sampler])

            The cycle can handle that using the `params` keyword:
            >>> cycle_with_parameters = FilesystemCycle(
            ...     metadata=metadata_0,
            ...     theorist=example_theorist,
            ...     experimentalist=example_experimentalist_with_parameters,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     raw_params={"experimentalist": {"uniform_random_sampler": {"n": 7}}}
            ... )
            >>> _ = list(takewhile(lambda c: len(c.state.conditions) < 1, cycle_with_parameters))
            >>> cycle_with_parameters.state.conditions[0].flatten()
            array([6.33661987, 7.34916618, 6.08596494, 2.28566582, 1.9553974 ,
                   5.80023149, 3.27007909])

            For the next cycle, if we wish, we can change the parameter value:
            >>> cycle_with_parameters.params["experimentalist"]["uniform_random_sampler"]\\
            ...     ["n"] = 2
            >>> _ = list(takewhile(lambda c: len(c.state.conditions) < 2, cycle_with_parameters))
            >>> cycle_with_parameters.state.conditions[1].flatten()
            array([10.5838232 ,  9.45666031])

            ### Accessing "Cycle Properties"

             Some experimentalists, experiment runners and theorists require access to the values
                created during the cycle execution, e.g. experimentalists which require access
                to the current best theory or the observed data. These data update each cycle, and
                so cannot easily be set using simple `params`.

            For this case, it is possible to use "cycle properties" in the `params` dictionary.
                These are the following strings, which will be replaced during execution by their
                respective current values:

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
            ...     sampled_conditions = random_sampler_rng.choice(
            ...         conditions, size=n, replace=False)
            ...     return sampled_conditions
            >>> def exclude_conditions(conditions, excluded_conditions):
            ...     remaining_conditions = list(
            ...         set(conditions) - set(excluded_conditions.flatten()))
            ...     return remaining_conditions
            >>> unobserved_data_experimentalist = make_pipeline([
            ...     metadata_1.independent_variables[0].allowed_values,
            ...     exclude_conditions,
            ...     custom_random_sampler
            ...     ]
            ... )
            >>> cycle_with_cycle_properties = FilesystemCycle(
            ...     metadata=metadata_1,
            ...     theorist=example_theorist,
            ...     experimentalist=unobserved_data_experimentalist,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     raw_params={
            ...         "experimentalist": {
            ...             "exclude_conditions": {"excluded_conditions": "%observations.ivs%"},
            ...             "custom_random_sampler": {"n": 1}
            ...         }
            ...     }
            ... )

            Now we can run the cycler to generate conditions and run experiments.
            The first time round, we have the full set of 10 possible conditions to select from,
            and we select "2" at random:
            >>> _ = cycle_with_cycle_properties.run(num_steps=3)
            >>> cycle_with_cycle_properties.state.conditions[-1]
            array([2])

            We can continue to run the cycler, each time we add more to the list of "excluded"
            options:
            >>> _ = cycle_with_cycle_properties.run(num_steps=15)
            >>> cycle_with_cycle_properties.state.conditions
            [array([2]), array([6]), array([5]), array([7]), array([3]), array([4])]

            By using the monitor callback, we can investigate what's going on with the cycle
            properties:
            >>> def observations_monitor(state):
            ...     if state[-1].kind == ResultKind.OBSERVATION:
            ...          print( _get_cycle_properties(state)["%observations.ivs%"].flatten())
            >>> cycle_with_cycle_properties.monitor = observations_monitor

            The monitor evaluates at the end of each cycle
            and shows that we've added a new observed IV each cycle
            >>> _ = cycle_with_cycle_properties.run(num_steps=3)
            [2. 6. 5. 7. 3. 4. 9.]
            >>> _ = cycle_with_cycle_properties.run(num_steps=3)
            [2. 6. 5. 7. 3. 4. 9. 0.]

            We deactivate the monitor by making it "None" again.
            >>> cycle_with_cycle_properties.monitor = None

            We can continue until we've sampled all of the options:
            >>> _ = cycle_with_cycle_properties.run(num_steps=6)
            >>> cycle_with_cycle_properties.state.conditions # doctest: +NORMALIZE_WHITESPACE
            [array([2]), array([6]), array([5]), array([7]), array([3]), \
            array([4]), array([9]), array([0]), array([8]), array([1])]

            If we try to evaluate it again, the experimentalist fails, as there aren't any more
            conditions which are available:
            >>> cycle_with_cycle_properties.run()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: a cannot be empty unless no samples are taken
        """

        self.experimentalist = wrap_experimentalist_autora_experimentalist_pipeline(
            experimentalist
        )
        self.experiment_runner = wrap_experiment_runner_synthetic_experiment(
            experiment_runner
        )
        self.theorist = wrap_theorist_scikit_learn(theorist)
        self.monitor = monitor
        if raw_params is None:
            raw_params = dict()
        self.raw_params = raw_params
        self.planner = planner

        self.state: ResultCollection = self._load_state(state, metadata, results)

        self.serializer = serializer

    @staticmethod
    def _load_state(
        state: Optional[Union[ResultCollectionSerializer, ResultCollection]],
        metadata: Optional[VariableCollection],
        results: Optional[
            Union[
                Sequence[Result],
                Result,
            ]
        ],
    ) -> ResultCollection:
        _state: ResultCollection

        if state is not None:
            assert metadata is None and results is None
            if isinstance(state, ResultCollectionSerializer):
                _state = state.load()
            elif isinstance(state, ResultCollection):
                _state = state

        else:
            assert state is None
            assert metadata is not None
            if results is None:
                _results = []
            elif isinstance(results, Result):
                _results = [results]
            elif isinstance(results, Sequence):
                _results = list(results)
            else:
                raise NotImplementedError(f"{results=} not supported")
            _state = ResultCollection(metadata=metadata, data=_results)

        return _state

    def run(self, num_steps: int = 1):
        for i in range(num_steps):
            next(self)
        return self

    def __iter__(self):
        return self

    def __next__(self):

        # Plan
        next_function = self.planner(self)

        # Execute
        result = next_function(self)

        # Store
        self.state.append(result)
        self.dump()

        # Monitor
        self._monitor_callback(self.state)

        return self

    def dump(self):
        if self.serializer is not None:
            self.serializer.dump(self.state)
        else:
            _logger.debug(f"{self.serializer=} must be set in order to dump")

    @property
    def params(self):
        """Returns the params dictionary, with "special" values like `theorist[-1]` resolved."""
        all_params = _resolve_cycle_properties(
            self.raw_params, _get_cycle_properties(self.state)
        )
        return all_params

    def _monitor_callback(self, data: ResultCollection):
        if self.monitor is not None:
            self.monitor(data)
