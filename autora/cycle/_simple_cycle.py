import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator

from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection


@dataclass(frozen=True)
class _SimpleCycleData:
    """An object passed between processing steps in the _SimpleCycle which holds all the
    data which can be updated."""

    # Static
    metadata: VariableCollection

    # Aggregates each cycle from the:
    # ... Experimentalist
    conditions: List[np.ndarray]
    # ... Experiment Runner
    observations: List[np.ndarray]
    # ... Theorist
    theories: List[BaseEstimator]


class _SimpleCycle:
    """

    Args:
        metadata:
        theorist:
        experimentalist:
        experiment_runner:

    Examples:

        Aim: Use the Cycle to recover a simple ground truth theory from noisy data.

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

        We initialize the Cycle with the metadata describing the domain of the theory,
        the theorist, experimentalist and experiment runner,
        as well as a monitor which will let us know which cycle we're currently on.
        >>> cycle = _SimpleCycle(
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ...     monitor=lambda data: print(f"Generated {len(data.theories)} theories"),
        ... )
        >>> cycle # doctest: +ELLIPSIS
        <_simple_cycle._SimpleCycle object at 0x...>

        We can run the cycle by calling the run method:
        >>> cycle.run(num_cycles=3)  # doctest: +ELLIPSIS
        Generated 1 theories
        Generated 2 theories
        Generated 3 theories
        <_simple_cycle._SimpleCycle object at 0x...>

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
        <_simple_cycle._SimpleCycle object at 0x...>

        >>> next(cycle) # doctest: +ELLIPSIS
        Generated 5 theories
        <_simple_cycle._SimpleCycle object at 0x...>

        >>> next(cycle) # doctest: +ELLIPSIS
        Generated 6 theories
        <_simple_cycle._SimpleCycle object at 0x...>

        We can continue to run the cycle as long as we like,
        with a simple arbitrary stopping condition like the number of theories generated:
        >>> from itertools import takewhile
        >>> _ = list(takewhile(lambda c: len(c.data.theories) < 9, cycle))
        Generated 7 theories
        Generated 8 theories
        Generated 9 theories

        ... or the precision (here we keep iterating while the difference between the gradients
        between the second-last and last cycle is larger than 1x10^-3).
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


        It's easy to pass parameters to the cycle components, if there are any needed.
        Here we have an experimentalist which takes a parameter:
        >>> uniform_random_rng = np.random.default_rng(180)
        >>> def uniform_random_sampler(num_samples):
        ...     return uniform_random_rng.uniform(low=0, high=11, size=num_samples)
        >>> example_experimentalist_with_parameters = make_pipeline([uniform_random_sampler])

        The cycle can handle that using the `params` keyword:
        >>> cycle_with_parameters = _SimpleCycle(
        ...     metadata=metadata_0,
        ...     theorist=example_theorist,
        ...     experimentalist=example_experimentalist_with_parameters,
        ...     experiment_runner=example_synthetic_experiment_runner,
        ...     params={"experimentalist": {"uniform_random_sampler": {"num_samples": 7}}}
        ... )
        >>> _ = cycle_with_parameters.run()
        >>> cycle_with_parameters.data.conditions[-1].flatten()
        array([6.33661987, 7.34916618, 6.08596494, 2.28566582, 1.9553974 ,
               5.80023149, 3.27007909])

        For the next cycle, if we wish, we can change the parameter value:
        >>> cycle_with_parameters.params["experimentalist"]["uniform_random_sampler"]\\
        ...     ["num_samples"] = 2
        >>> _ = cycle_with_parameters.run()
        >>> cycle_with_parameters.data.conditions[-1].flatten()
        array([10.5838232 ,  9.45666031])


         In the case we have an experimentalist which needs variables such as
        - the current best theory
        - all the existing observational data
        to generate its next values, we can use magic parameters:
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
        >>> cycle_with_magic_parameters = _SimpleCycle(
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
        >>> _ = cycle_with_magic_parameters.run()
        >>> cycle_with_magic_parameters.data.conditions[-1]
        array([2])

        We can continue to run the cycler, each time we add more to the list of "excluded" options:
        >>> _ = cycle_with_magic_parameters.run(num_cycles=5)
        >>> cycle_with_magic_parameters.data.conditions
        [array([2]), array([6]), array([5]), array([7]), array([3]), array([4])]

        By using the monitor callback, we can investigate what's going on with the magic parameters:
        >>> cycle_with_magic_parameters.monitor = lambda data: print(
        ...     _get_cycle_properties(data)["%observations.ivs%"].flatten()
        ... )

        The monitor evaluates at the end of each cycle
        and shows that we've added a new observed IV each step
        >>> _ = cycle_with_magic_parameters.run()
        [2. 6. 5. 7. 3. 4. 9.]
        >>> _ = cycle_with_magic_parameters.run()
        [2. 6. 5. 7. 3. 4. 9. 0.]

        We deactivate the monitor by making it "None" again.
        >>> cycle_with_magic_parameters.monitor = None

        We can continue until we've sampled all of the options:
        >>> _ = cycle_with_magic_parameters.run(num_cycles=2)
        >>> cycle_with_magic_parameters.data.conditions # doctest: +NORMALIZE_WHITESPACE
        [array([2]), array([6]), array([5]), array([7]), array([3]), \
        array([4]), array([9]), array([0]), array([8]), array([1])]

        If we try to evaluate it again, the experimentalist fails, as there aren't any more
        conditions which are available:
        >>> cycle_with_magic_parameters.run()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: a cannot be empty unless no samples are taken

    """

    def __init__(
        self,
        metadata: VariableCollection,
        theorist,
        experimentalist,
        experiment_runner,
        monitor: Optional[Callable[[_SimpleCycleData], None]] = None,
        params: Optional[Dict] = None,
    ):

        self.theorist = theorist
        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner
        self.monitor = monitor
        if params is None:
            params = dict()
        self.params = params

        self.data = _SimpleCycleData(
            metadata=metadata,
            conditions=[],
            observations=[],
            theories=[],
        )

    def run(self, num_cycles: int = 1):
        for i in range(num_cycles):
            next(self)
        return self

    def __next__(self):
        assert (
            "experiment_runner" not in self.params
        ), "experiment_runner cannot yet accept cycle parameters"
        assert (
            "theorist" not in self.params
        ), "theorist cannot yet accept cycle parameters"

        data = self.data
        params_with_magics = _resolve_cycle_properties(
            self.params, _get_cycle_properties(self.data)
        )

        data = self._experimentalist_callback(
            self.experimentalist,
            data,
            params_with_magics.get("experimentalist", dict()),
        )
        data = self._experiment_runner_callback(self.experiment_runner, data)
        data = self._theorist_callback(self.theorist, data)
        self._monitor_callback(data)
        self.data = data

        return self

    def __iter__(self):
        return self

    @staticmethod
    def _experimentalist_callback(
        experimentalist: Pipeline, data_in: _SimpleCycleData, params: dict
    ):
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
        data_out = replace(
            data_in,
            conditions=data_in.conditions + [new_conditions_array],
        )
        return data_out

    @staticmethod
    def _experiment_runner_callback(
        experiment_runner: Callable, data_in: _SimpleCycleData
    ):
        x = data_in.conditions[-1]
        y = experiment_runner(x)
        new_observations = np.column_stack([x, y])
        data_out = replace(
            data_in, observations=data_in.observations + [new_observations]
        )
        return data_out

    @staticmethod
    def _theorist_callback(theorist, data_in: _SimpleCycleData):
        all_observations = np.row_stack(data_in.observations)
        n_xs = len(
            data_in.metadata.independent_variables
        )  # The number of independent variables
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        new_theorist = copy.copy(theorist)
        new_theorist.fit(x, y)
        data_out = replace(
            data_in,
            theories=data_in.theories + [new_theorist],
        )
        return data_out

    def _monitor_callback(self, data: _SimpleCycleData):
        if self.monitor is not None:
            self.monitor(data)


def _resolve_cycle_properties(params: Dict, cycle_properties: Mapping):
    """
    Resolve "magic values" inside a nested dictionary.

    In this context, a "magic value" is a string which is meant to be replaced by a different value
    before the dictionary is accessed.

    Args:
        params: a (nested) dictionary of keys and values, where some values might be
            "cycle property names"
        cycle_properties: a dictionary of "cycle property names" and their "real values"

    Returns: a (nested) dictionary where "cycle property names" are replaced by the "real values"

    Examples:

        >>> params_0 = {"key": "%foo%"}
        >>> cycle_properties_0 = {"%foo%": 180}
        >>> _resolve_cycle_properties(params_0, cycle_properties_0)
        {'key': 180}

        >>> params_1 = {"key": "%bar%", "nested_dict": {"inner_key": "%foobar%"}}
        >>> cycle_properties_1 = {"%bar%": 1, "%foobar%": 2}
        >>> _resolve_cycle_properties(params_1, cycle_properties_1)
        {'key': 1, 'nested_dict': {'inner_key': 2}}

    """
    params_ = copy.copy(params)
    for key, value in params_.items():
        if isinstance(value, dict):
            params_[key] = _resolve_cycle_properties(value, cycle_properties)
        elif (
            isinstance(value, str) and value in cycle_properties
        ):  # value is a key in the cycle_properties dictionary
            params_[key] = cycle_properties[value]
        else:
            pass  # no change needed

    return params_


class LazyDict(Mapping):
    """Inspired by https://gist.github.com/gyli/9b50bb8537069b4e154fec41a4b5995a"""

    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func = self._raw_dict.__getitem__(key)
        return func()

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def _get_cycle_properties(data: _SimpleCycleData):
    """
    Examples:
        Even with an empty data object, we can initialize the dictionary,
        >>> cycle_properties = _get_cycle_properties(_SimpleCycleData(metadata=None, conditions=[],
        ...     observations=[], theories=[]))

        ... but it will raise an exception if a value isn't yet available when we try to use it
        >>> cycle_properties["%theories[-1]%"] # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: list index out of range

        Nevertheless, we can iterate through its keys no problem:
        >>> [key for key in cycle_properties.keys()] # doctest: +NORMALIZE_WHITESPACE
        ['%observations.ivs[-1]%', '%observations.dvs[-1]%', '%observations.ivs%',
        '%observations.dvs%', '%theories[-1]%', '%theories%']

    """

    n_ivs = len(data.metadata.independent_variables)
    n_dvs = len(data.metadata.dependent_variables)
    cycle_property_dict = LazyDict(
        {
            "%observations.ivs[-1]%": lambda: data.observations[-1][:, 0:n_ivs],
            "%observations.dvs[-1]%": lambda: data.observations[-1][:, n_ivs:],
            "%observations.ivs%": lambda: np.row_stack(
                [np.empty([0, n_ivs + n_dvs])] + data.observations
            )[:, 0:n_ivs],
            "%observations.dvs%": lambda: np.row_stack(data.observations)[:, n_ivs:],
            "%theories[-1]%": lambda: data.theories[-1],
            "%theories%": lambda: data.theories,
        }
    )
    return cycle_property_dict
