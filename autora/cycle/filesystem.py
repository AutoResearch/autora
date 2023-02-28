""" ASync Cycle """

import copy
import logging
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import yaml

import autora.utils.YAMLSerializer as YAMLSerializer
from autora.cycle.simple import _get_cycle_properties, _resolve_cycle_properties
from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)


class ResultType(Enum):
    CONDITION = auto()
    OBSERVATION = auto()
    THEORY = auto()

    def __str__(self):
        return f"{self.name}"


class ResultContainer(NamedTuple):
    data: Optional[Any]
    kind: Optional[ResultType]


def _get_all_of_kind(data: Sequence[ResultContainer], kind: ResultType):
    filtered_data = [
        c.data for c in data if ((c.kind is not None) and (c.kind == kind))
    ]
    return filtered_data


@dataclass
class FilesystemCycleDataCollection:
    metadata: VariableCollection
    data: List[ResultContainer]

    @property
    def conditions(self):
        return _get_all_of_kind(self.data, ResultType.CONDITION)

    @property
    def observations(self):
        return _get_all_of_kind(self.data, ResultType.OBSERVATION)

    @property
    def theories(self):
        return _get_all_of_kind(self.data, ResultType.THEORY)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> ResultContainer:
        return self.data[index]

    def append(self, item: ResultContainer):
        self.data.append(item)
        return


def _dumper(data_collection: FilesystemCycleDataCollection, path: Path):
    """


    Args:
        data_collection:
        path: a directory

    Returns:

    Examples:
        First, we need to initialize a FilesystemCycleDataCollection. This is usually handled
        by the cycle itself. We start with a data collection as it would be at the very start of
        an experiment, with just a VariableCollection.
        >>> metadata = VariableCollection()
        >>> c = FilesystemCycleDataCollection(metadata=metadata, data=[])
        >>> c # doctest: +NORMALIZE_WHITESPACE
        FilesystemCycleDataCollection(metadata=VariableCollection(independent_variables=[], \
        dependent_variables=[], covariates=[]), data=[])

        Now we can serialize the data collection using _dumper. We define a helper function for
        demonstration purposes.
        >>> import tempfile
        >>> import os
        >>> def dump_and_list(data, cat=False):
        ...     with tempfile.TemporaryDirectory() as d:
        ...         _dumper(c, d)
        ...         print(sorted(os.listdir(d)))

        Each immutable part gets its own file.
        >>> dump_and_list(c)
        ['metadata.yaml']

        The next step is to plan the first observations by defining experimental conditions.
        Thes are appended as a ResultContainer with the correct metadata.
        >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
        >>> c.append(ResultContainer(x, ResultType.CONDITION))

        If we dump and list again, we see that the new data are included as a new file in the same
        directory.
        >>> dump_and_list(c)
        ['00000000.yaml', 'metadata.yaml']

        Then, once we've gathered real data, we dump these too:
        >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
        >>> c.append(ResultContainer(np.column_stack([x, y]), ResultType.OBSERVATION))
        >>> dump_and_list(c)
        ['00000000.yaml', '00000001.yaml', 'metadata.yaml']

        We can also include a theory in the dump. The theory is saved as a pickle file by default
        >>> from sklearn.linear_model import LinearRegression
        >>> estimator = LinearRegression().fit(x, y)
        >>> c.append(ResultContainer(estimator, ResultType.THEORY))
        >>> dump_and_list(c)
        ['00000000.yaml', '00000001.yaml', '00000002.pickle', 'metadata.yaml']

    """
    if Path(path).exists():
        assert Path(path).is_dir(), "Can't support individual files now."
    else:
        Path(path).mkdir()

    metadata_extension = "yaml"
    metadata_str = yaml.dump(data_collection.metadata)
    with open(Path(path, f"metadata.{metadata_extension}"), "w+") as f:
        f.write(metadata_str)

    for i, container in enumerate(data_collection.data):
        extension, serializer, mode = {
            None: ("yaml", YAMLSerializer, "w+"),
            ResultType.CONDITION: ("yaml", YAMLSerializer, "w+"),
            ResultType.OBSERVATION: ("yaml", YAMLSerializer, "w+"),
            ResultType.THEORY: ("pickle", pickle, "w+b"),
        }[container.kind]
        filename = f"{str(i).rjust(8, '0')}.{extension}"
        with open(Path(path, filename), mode) as f:
            serializer.dump(container, f)


def _loader(path: Path):
    """

    Examples:
        First, we need to initialize a FilesystemCycleDataCollection. This is usually handled
        by the cycle itself. We construct a full set of results:
        >>> from sklearn.linear_model import LinearRegression
        >>> import tempfile
        >>> metadata = VariableCollection()
        >>> c = FilesystemCycleDataCollection(metadata=metadata, data=[])
        >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
        >>> c.append(ResultContainer(x, ResultType.CONDITION))
        >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
        >>> c.append(ResultContainer(np.column_stack([x, y]), ResultType.OBSERVATION))
        >>> estimator = LinearRegression().fit(x, y)
        >>> c.append(ResultContainer(estimator, ResultType.THEORY))

        Now we can serialize the data using _dumper, and reload the data using _loader:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _dumper(c, d)
        ...     e = _loader(d)

        We can now compare the dumped object "c" with the reloaded object "e". The data arrays
        should be equal, and the theories should
        >>> assert e.metadata == c.metadata
        >>> for e_i, c_i in zip(e, c):
        ...     assert isinstance(e_i.data, type(c_i.data)) # Types match
        ...     if e_i.kind in (ResultType.CONDITION, ResultType.OBSERVATION):
        ...         np.testing.assert_array_equal(e_i.data, c_i.data) # two numpy arrays
        ...     if e_i.kind == ResultType.THEORY:
        ...         np.testing.assert_array_equal(e_i.data.coef_, c_i.data.coef_) # two estimators

    """
    assert Path(path).is_dir(), f"{path=} must be a directory."
    metadata = None
    data = []

    for file in sorted(Path(path).glob("*")):
        serializer, mode = {".yaml": (YAMLSerializer, "r"), ".pickle": (pickle, "rb")}[
            file.suffix
        ]
        with open(file, mode) as f:
            loaded_object = serializer.load(f)
        if isinstance(loaded_object, VariableCollection):
            metadata = loaded_object
        else:
            data.append(loaded_object)

    assert isinstance(metadata, VariableCollection)
    data_collection = FilesystemCycleDataCollection(metadata=metadata, data=data)

    return data_collection


class FilesystemCycle:
    def __init__(
        self,
        theorist,
        experimentalist,
        experiment_runner,
        monitor: Optional[Callable[[FilesystemCycleDataCollection], None]] = None,
        metadata: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        data: Optional[
            Union[Path, str, Sequence[ResultContainer], ResultContainer]
        ] = None,
        path: Optional[Path] = None,
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
            >>> cycle = FilesystemCycle(
            ...     metadata=metadata_0,
            ...     theorist=example_theorist,
            ...     experimentalist=example_experimentalist,
            ...     experiment_runner=example_synthetic_experiment_runner,
            ...     monitor=
            ...         lambda data: print(f"Generated {len(data)}-th datum, a new {data[-1].kind}")
            ... )
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
            >>> _ = list(takewhile(lambda c: len(c.data.theories) < 9, cycle)) # doctest: +ELLIPSIS
            Generated 13-th datum, a new CONDITION
            ...
            Generated 27-th datum, a new THEORY

            ... or the precision (here we keep iterating while the difference between the gradients
            of the second-last and last cycle is larger than 1x10^-3).
            >>> _ = list(
            ...         takewhile(
            ...             lambda c: np.abs(c.data.theories[-1].coef_.item() -
            ...                            c.data.theories[-2].coef_.item()) > 1e-3,
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
            ...     params={"experimentalist": {"uniform_random_sampler": {"n": 7}}}
            ... )
            >>> _ = list(takewhile(lambda c: len(c.data.conditions) < 1, cycle_with_parameters))
            >>> cycle_with_parameters.data.conditions[0].flatten()
            array([6.33661987, 7.34916618, 6.08596494, 2.28566582, 1.9553974 ,
                   5.80023149, 3.27007909])

            For the next cycle, if we wish, we can change the parameter value:
            >>> cycle_with_parameters.params["experimentalist"]["uniform_random_sampler"]\\
            ...     ["n"] = 2
            >>> _ = list(takewhile(lambda c: len(c.data.conditions) < 2, cycle_with_parameters))
            >>> cycle_with_parameters.data.conditions[1].flatten()
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
            ...     params={
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
            >>> cycle_with_cycle_properties.data.conditions[-1]
            array([2])

            We can continue to run the cycler, each time we add more to the list of "excluded"
            options:
            >>> _ = cycle_with_cycle_properties.run(num_steps=15)
            >>> cycle_with_cycle_properties.data.conditions
            [array([2]), array([6]), array([5]), array([7]), array([3]), array([4])]

            By using the monitor callback, we can investigate what's going on with the cycle
            properties:
            >>> def observations_monitor(data):
            ...     if data[-1].kind == ResultType.OBSERVATION:
            ...          print( _get_cycle_properties(data)["%observations.ivs%"].flatten())
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
            >>> cycle_with_cycle_properties.data.conditions # doctest: +NORMALIZE_WHITESPACE
            [array([2]), array([6]), array([5]), array([7]), array([3]), \
            array([4]), array([9]), array([0]), array([8]), array([1])]

            If we try to evaluate it again, the experimentalist fails, as there aren't any more
            conditions which are available:
            >>> cycle_with_cycle_properties.run()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: a cannot be empty unless no samples are taken
        """

        self.experimentalist = experimentalist
        self.experiment_runner = experiment_runner
        self.theorist = theorist
        self.monitor = monitor
        if params is None:
            params = dict()
        self.params = params
        self.path = path

        # Load the data
        self.data = self._load_data(data, metadata)

    @staticmethod
    def _load_data(
        data: Optional[
            Union[
                Path,
                str,
                FilesystemCycleDataCollection,
                Sequence[ResultContainer],
                ResultContainer,
            ]
        ],
        metadata: Optional[VariableCollection],
    ) -> FilesystemCycleDataCollection:
        if isinstance(data, Path):
            _data = _loader(data)
        elif isinstance(data, str):
            _data = _loader(Path(data))
        elif isinstance(data, FilesystemCycleDataCollection):
            _data = data
        elif isinstance(data, Sequence):
            assert metadata is not None
            assert isinstance(data, List)
            _data = FilesystemCycleDataCollection(metadata=metadata, data=data)
        elif isinstance(data, ResultContainer):
            assert metadata is not None
            _data = FilesystemCycleDataCollection(metadata=metadata, data=[data])
        elif data is None:
            assert metadata is not None
            _data = FilesystemCycleDataCollection(metadata=metadata, data=[])
        else:
            raise ValueError(f"{data=}, {metadata=} missing something")
        return _data

    def run(self, num_steps: int = 1):
        for i in range(num_steps):
            next(self)
        return self

    def __iter__(self):
        return self

    def __next__(self):

        next_function = self._plan_next_step()

        # Execute
        result = next_function()

        # Store
        self.data.append(result)
        if self.path is not None:
            self.dump()

        # Monitor
        self._monitor_callback(self.data)

        return self

    def dump(self, path=None):
        if path is None:
            path = self.path
        _dumper(data_collection=self.data, path=path)

    def _plan_next_step(self):  # TODO: move the business logic to a separate function
        all_params = _resolve_cycle_properties(
            self.params, _get_cycle_properties(self.data)
        )

        curried_experimentalist = partial(
            self._experimentalist_callback,
            experimentalist=self.experimentalist,
            data_in=self.data,
            params=all_params.get("experimentalist", dict()),
        )
        curried_theorist = partial(
            self._theorist_callback,
            theorist=self.theorist,
            data_in=self.data,
            params=all_params.get("theorist", dict()),
        )
        curried_experiment_runner = partial(
            self._experiment_runner_callback,
            experiment_runner=self.experiment_runner,
            data_in=self.data,
            params=all_params.get("experiment_runner", dict()),
        )

        curried_callback = last_result_kind_planner(
            state=self.data,
            mapping={
                None: curried_experimentalist,
                ResultType.THEORY: curried_experimentalist,
                ResultType.CONDITION: curried_experiment_runner,
                ResultType.OBSERVATION: curried_theorist,
            },
        )

        return curried_callback

    @staticmethod
    def _experimentalist_callback(
        experimentalist: Pipeline, data_in: FilesystemCycleDataCollection, params: dict
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

        result = ResultContainer(data=new_conditions_array, kind=ResultType.CONDITION)

        return result

    @staticmethod
    def _experiment_runner_callback(
        experiment_runner: Callable,
        data_in: FilesystemCycleDataCollection,
        params: dict,
    ):
        x = data_in.conditions[-1]
        y = experiment_runner(x, **params)
        new_observations = np.column_stack([x, y])

        result = ResultContainer(data=new_observations, kind=ResultType.OBSERVATION)

        return result

    @staticmethod
    def _theorist_callback(
        theorist, data_in: FilesystemCycleDataCollection, params: dict
    ):
        all_observations = np.row_stack(data_in.observations)
        n_xs = len(
            data_in.metadata.independent_variables
        )  # The number of independent variables
        x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
        if y.shape[1] == 1:
            y = y.ravel()
        new_theorist = copy.deepcopy(theorist)
        new_theorist.fit(x, y, **params)

        result = ResultContainer(data=new_theorist, kind=ResultType.THEORY)

        return result

    def _monitor_callback(self, data: FilesystemCycleDataCollection):
        if self.monitor is not None:
            self.monitor(data)


def last_result_kind_planner(
    state: FilesystemCycleDataCollection, mapping: Dict[Optional[ResultType], Callable]
):
    try:
        last_result = state[-1]
    except IndexError:
        last_result = ResultContainer(None, None)
    next = mapping[last_result.kind]
    return next
