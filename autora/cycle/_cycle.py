import copy
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List

import numpy as np
from sklearn.base import BaseEstimator

from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection


@dataclass(frozen=True)
class _SimpleCycleRunCollection:
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
    def __init__(
        self,
        metadata,
        theorist,
        experimentalist,
        experiment_runner,
        max_cycle_count,
        cycle_count=0,
        monitors=None,
    ):
        """

        Args:
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

            We initialize the Cycle with the theorist, experimentalist and experiment runner,
            and define the maximum cycle count.
            >>> cycle = _SimpleCycle(
            ...     metadata=study_metadata,
            ...     theorist=example_theorist,
            ...     experimentalist=example_experimentalist,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     max_cycle_count=3,
            ...     monitors=[lambda data: print(f"Finished cycle {data.cycle_count}")],
            ... )
            >>> cycle # doctest: +ELLIPSIS
            <_cycle._SimpleCycle object at 0x...>

            We can run the cycle by calling the run method:
            >>> cycle.run()
            Finished cycle 1
            Finished cycle 2
            Finished cycle 3

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

        """

        self.theorist = theorist
        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner
        self.monitors = monitors

        self.data = _SimpleCycleRunCollection(
            metadata=metadata,
            max_cycle_count=max_cycle_count,
            cycle_count=cycle_count,
            conditions=[],
            observations=[],
            theories=[],
        )

    def run(self):
        data = self.data

        while not (self._stopping_condition(data)):
            data = self._experimentalist_callback(self.experimentalist, data)
            data = self._experiment_runner_callback(self.experiment_runner, data)
            data = self._theorist_callback(self.theorist, data)
            self._monitor_callback(data)

        self.data = data

    @staticmethod
    def _experimentalist_callback(
        experimentalist: Pipeline, data_in: _SimpleCycleRunCollection
    ):
        new_conditions = experimentalist()
        if isinstance(new_conditions, Iterable):
            # If the pipeline gives us an iterable, we need to make it into a concrete array.
            # We can't move this logic to the Pipeline, because the pipeline doesn't know whether
            # it's within another pipeline and whether it should convert the iterable to a
            # concrete array.
            new_conditions_values = list(new_conditions)
            new_conditions_array = np.array(new_conditions_values)

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
        experiment_runner: Callable, data_in: _SimpleCycleRunCollection
    ):
        x = data_in.conditions[-1]
        y = experiment_runner(x)

        new_observations = np.column_stack([x, y])

        data_out = replace(
            data_in, observations=data_in.observations + [new_observations]
        )
        return data_out

    @staticmethod
    def _theorist_callback(theorist, data_in: _SimpleCycleRunCollection):
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

    def _monitor_callback(self, data: _SimpleCycleRunCollection):
        for m in self.monitors:
            m(data)
