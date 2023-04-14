import pickle
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Generic, Mapping, NamedTuple, Union

import numpy as np

from autora.controller.protocol import ResultKind, State, SupportsLoadDump
from autora.controller.serializer import yaml_ as YAMLSerializer
from autora.controller.state import History


class _DumpSpec(NamedTuple):
    extension: str
    serializer: SupportsLoadDump
    mode: str


class _LoadSpec(NamedTuple):
    serializer: SupportsLoadDump
    mode: str


class StateSerializer(Generic[State]):
    @abstractmethod
    def load(self) -> State:
        ...

    @abstractmethod
    def dump(self, ___state: State):
        ...


class HistorySerializer(StateSerializer[History]):
    """Serializes and deserializes History objects."""

    def __init__(
        self,
        path: Path,
    ):
        self.path = path
        self._check_path()

        self._result_kind_serializer_mapping: Mapping[
            Union[None, ResultKind], _DumpSpec
        ] = {
            None: _DumpSpec("yaml", YAMLSerializer, "w+"),
            ResultKind.METADATA: _DumpSpec("yaml", YAMLSerializer, "w+"),
            ResultKind.PARAMS: _DumpSpec("yaml", YAMLSerializer, "w+"),
            ResultKind.CONDITION: _DumpSpec("yaml", YAMLSerializer, "w+"),
            ResultKind.OBSERVATION: _DumpSpec("yaml", YAMLSerializer, "w+"),
            ResultKind.THEORY: _DumpSpec("pickle", pickle, "w+b"),
        }

        self._extension_loader_mapping: Mapping[str, _LoadSpec] = {
            ".yaml": _LoadSpec(YAMLSerializer, "r"),
            ".pickle": _LoadSpec(pickle, "rb"),
        }

    def dump(self, data_collection: History):
        """

        Args:
            data_collection:
            path: a directory

        Returns:

        Examples:
            First, we need to initialize a FilesystemCycleDataCollection. This is usually handled
            by the cycle itself. We start with a data collection as it would be at the very start of
            an experiment, with just a VariableCollection.
            >>> from autora.controller.state.history import History
            >>> c = History()
            >>> c  #doctest: +NORMALIZE_WHITESPACE
            History([])

            Now we can serialize the data collection using _dumper. We define a helper function for
            demonstration purposes.
            >>> import tempfile
            >>> import os
            >>> def dump_and_list(data, cat=False):
            ...     with tempfile.TemporaryDirectory() as d:
            ...         s = HistorySerializer(d)
            ...         s.dump(data)
            ...         print(sorted(os.listdir(d)))

            >>> dump_and_list(c)
            []

            Each immutable part gets its own file.
            >>> from autora.variable import VariableCollection
            >>> c = c.update(metadata=[VariableCollection()])
            >>> dump_and_list(c)
            ['00000000-METADATA.yaml']

            The next step is to plan the first observations by defining experimental conditions.
            Thes are appended as a Result with the correct metadata.
            >>> import numpy as np
            >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
            >>> c = c.update(conditions=[x])

            If we dump and list again, we see that the new data are  included as a new file in
            the same directory.
            >>> dump_and_list(c)
            ['00000000-METADATA.yaml', '00000001-CONDITION.yaml']

            Then, once we've gathered real data, we dump these too:
            >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
            >>> c = c.update(observations=[np.column_stack([x, y])])
            >>> dump_and_list(c)
            ['00000000-METADATA.yaml', '00000001-CONDITION.yaml', '00000002-OBSERVATION.yaml']

            We can also include a theory in the dump.
            The theory is saved as a pickle file by default.
            >>> from sklearn.linear_model import LinearRegression
            >>> estimator = LinearRegression().fit(x, y)
            >>> c = c.update(theories=[estimator])
            >>> dump_and_list(c)  # doctest: +NORMALIZE_WHITESPACE
            ['00000000-METADATA.yaml', '00000001-CONDITION.yaml', '00000002-OBSERVATION.yaml',
             '00000003-THEORY.pickle']


        """
        path = self.path
        self._check_path()

        for i, container in enumerate(data_collection.history):
            extension, serializer, mode = self._result_kind_serializer_mapping[
                container.kind
            ]

            assert isinstance(serializer, SupportsLoadDump)
            filename = f"{str(i).rjust(8, '0')}-{container.kind}.{extension}"
            with open(Path(path, filename), mode) as f:
                serializer.dump(container, f)

    def load(self) -> History:
        """

        Examples:
            First, we need to initialize a FilesystemCycleDataCollection. This is usually handled
            by the cycle itself. We construct a full set of results:
            >>> from sklearn.linear_model import LinearRegression
            >>> from autora.variable import VariableCollection
            >>> import numpy as np
            >>> from autora.controller.state.history import History
            >>> import tempfile
            >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
            >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
            >>> estimator = LinearRegression().fit(x, y)
            >>> c = History(metadata=VariableCollection(), conditions=[x],
            ...     observations=[np.column_stack([x, y])], theories=[estimator])

            Now we can serialize the data using _dumper, and reload the data using _loader:
            >>> with tempfile.TemporaryDirectory() as d:
            ...     s = HistorySerializer(d)
            ...     s.dump(c)
            ...     e = s.load()

            We can now compare the dumped object "c" with the reloaded object "e". The data arrays
            should be equal, and the theories should
            >>> from autora.controller.protocol import ResultKind
            >>> for e_i, c_i in zip(e.history, c.history):
            ...     assert isinstance(e_i.data, type(c_i.data)) # Types match
            ...     if e_i.kind in (ResultKind.CONDITION, ResultKind.OBSERVATION):
            ...         np.testing.assert_array_equal(e_i.data, c_i.data) # two numpy arrays
            ...     if e_i.kind == ResultKind.THEORY:
            ...         np.testing.assert_array_equal(e_i.data.coef_, c_i.data.coef_) # 2 estimators


            We can also have the function load a subclass of the History object, or something
            else which supports its interface:
            >>> class DerivedHistory(History):
            ...     pass
            >>> with tempfile.TemporaryDirectory() as d:
            ...     s = HistorySerializer(d)
            ...     s.dump(c)
            ...     f = s.load(cls=DerivedHistory)
            >>> f  # doctest: +ELLIPSIS
            DerivedHistory(...)

        """
        path = self.path
        assert Path(path).is_dir(), f"{path=} must be a directory."
        data = []

        for file in sorted(Path(path).glob("*")):
            serializer, mode = self._extension_loader_mapping[file.suffix]
            with open(file, mode) as f:
                loaded_object = serializer.load(f)
                data.append(loaded_object)

        data_collection = History(history=data)

        return data_collection

    def _check_path(self):
        """Ensure the path exists and is the right type."""
        if Path(self.path).exists():
            assert Path(self.path).is_dir(), "Can't support individual files now."
        else:
            Path(self.path).mkdir()
