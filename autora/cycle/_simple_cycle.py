import copy
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Optional

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
        >>> study_metadata = VariableCollection(
        ...    independent_variables=[Variable(name="x1", allowed_values=range(11))],
        ...    dependent_variables=[Variable(name="y", value_range=(-20, 20))],
        ...    )

        The experimentalist is used to propose experiments.
        Since the space of values is so restricted, we can just sample them all each time.
        >>> from autora.experimentalist.pipeline import make_pipeline
        >>> example_experimentalist = make_pipeline(
        ...     [study_metadata.independent_variables[0].allowed_values])

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
        ...     metadata=study_metadata,
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




    """

    def __init__(
        self,
        metadata: VariableCollection,
        theorist,
        experimentalist,
        experiment_runner,
        monitor: Optional[Callable[[_SimpleCycleData], None]] = None,
    ):

        self.theorist = theorist
        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner
        self.monitor = monitor

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
        data = self.data
        data = self._experimentalist_callback(self.experimentalist, data)
        data = self._experiment_runner_callback(self.experiment_runner, data)
        data = self._theorist_callback(self.theorist, data)
        self._monitor_callback(data)
        self.data = data
        return self

    def __iter__(self):
        return self

    @staticmethod
    def _experimentalist_callback(experimentalist: Pipeline, data_in: _SimpleCycleData):
        new_conditions = experimentalist()
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
