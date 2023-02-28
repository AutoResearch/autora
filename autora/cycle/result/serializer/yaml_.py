import pickle
from pathlib import Path

from autora.cycle.result import ResultCollection, ResultKind
from autora.cycle.result.serializer import ResultCollectionSerializer
from autora.utils import YAMLSerializer as YAMLSerializer
from autora.variable import VariableCollection


class YAMLResultCollectionSerializer(ResultCollectionSerializer):
    def __init__(self, path: Path):
        self.path = path
        self._check_path()

    def dump(self, data_collection: ResultCollection):
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
            >>> from autora.cycle.result import Result, ResultKind, ResultCollection
            >>> c = ResultCollection(metadata=metadata, data=[])
            >>> c  #doctest: +NORMALIZE_WHITESPACE
            ResultCollection(metadata=VariableCollection(independent_variables=[], \
            dependent_variables=[], covariates=[]), data=[])

            Now we can serialize the data collection using _dumper. We define a helper function for
            demonstration purposes.
            >>> import tempfile
            >>> import os
            >>> def dump_and_list(data, cat=False):
            ...     with tempfile.TemporaryDirectory() as d:
            ...         dump(c, d)
            ...         print(sorted(os.listdir(d)))

            Each immutable part gets its own file.
            >>> dump_and_list(c)
            ['metadata.yaml']

            The next step is to plan the first observations by defining experimental conditions.
            Thes are appended as a Result with the correct metadata.
            >>> import numpy as np
            >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
            >>> c.append(Result(x, ResultKind.CONDITION))

            If we dump and list again, we see that the new data are  included as a new file in
            the same directory.
            >>> dump_and_list(c)
            ['00000000.yaml', 'metadata.yaml']

            Then, once we've gathered real data, we dump these too:
            >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
            >>> c.append(Result(np.column_stack([x, y]), ResultKind.OBSERVATION))
            >>> dump_and_list(c)
            ['00000000.yaml', '00000001.yaml', 'metadata.yaml']

            We can also include a theory in the dump.
            The theory is saved as a pickle file by default.
            >>> from sklearn.linear_model import LinearRegression
            >>> estimator = LinearRegression().fit(x, y)
            >>> c.append(Result(estimator, ResultKind.THEORY))
            >>> dump_and_list(c)
            ['00000000.yaml', '00000001.yaml', '00000002.pickle', 'metadata.yaml']

        """
        path = self.path
        self._check_path()

        metadata_extension = "yaml"
        with open(Path(path, f"metadata.{metadata_extension}"), "w+") as f:
            YAMLSerializer.dump(data_collection.metadata, f)

        for i, container in enumerate(data_collection.data):
            extension, serializer, mode = {
                None: ("yaml", YAMLSerializer, "w+"),
                ResultKind.CONDITION: ("yaml", YAMLSerializer, "w+"),
                ResultKind.OBSERVATION: ("yaml", YAMLSerializer, "w+"),
                ResultKind.THEORY: ("pickle", pickle, "w+b"),
            }[container.kind]
            filename = f"{str(i).rjust(8, '0')}.{extension}"
            with open(Path(path, filename), mode) as f:
                serializer.dump(container, f)

    def load(self):
        """

        Examples:
            First, we need to initialize a FilesystemCycleDataCollection. This is usually handled
            by the cycle itself. We construct a full set of results:
            >>> from sklearn.linear_model import LinearRegression
            >>> import numpy as np
            >>> from autora.cycle.result import Result, ResultKind
            >>> import tempfile
            >>> metadata = VariableCollection()
            >>> c = ResultCollection(metadata=metadata, data=[])
            >>> x = np.linspace(-2, 2, 10).reshape(-1, 1) * np.pi
            >>> c.append(Result(x, ResultKind.CONDITION))
            >>> y = 3. * x + 0.1 * np.sin(x - 0.1) - 2.
            >>> c.append(Result(np.column_stack([x, y]), ResultKind.OBSERVATION))
            >>> estimator = LinearRegression().fit(x, y)
            >>> c.append(Result(estimator, ResultKind.THEORY))

            Now we can serialize the data using _dumper, and reload the data using _loader:
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dump(c, d)
            ...     e = load(d)

            We can now compare the dumped object "c" with the reloaded object "e". The data arrays
            should be equal, and the theories should
            >>> assert e.metadata == c.metadata
            >>> for e_i, c_i in zip(e, c):
            ...     assert isinstance(e_i.data, type(c_i.data)) # Types match
            ...     if e_i.kind in (ResultKind.CONDITION, ResultKind.OBSERVATION):
            ...         np.testing.assert_array_equal(e_i.data, c_i.data) # two numpy arrays
            ...     if e_i.kind == ResultKind.THEORY:
            ...         np.testing.assert_array_equal(e_i.data.coef_, c_i.data.coef_) # 2 estimators

        """
        path = self.path
        assert Path(path).is_dir(), f"{path=} must be a directory."
        metadata = None
        data = []

        for file in sorted(Path(path).glob("*")):
            serializer, mode = {
                ".yaml": (YAMLSerializer, "r"),
                ".pickle": (pickle, "rb"),
            }[file.suffix]
            with open(file, mode) as f:
                loaded_object = serializer.load(f)
            if isinstance(loaded_object, VariableCollection):
                metadata = loaded_object
            else:
                data.append(loaded_object)

        assert isinstance(metadata, VariableCollection)
        data_collection = ResultCollection(metadata=metadata, data=data)

        return data_collection

    def _check_path(self):
        """Ensure the path exists and is the right type."""
        if Path(self.path).exists():
            assert Path(self.path).is_dir(), "Can't support individual files now."
        else:
            Path(self.path).mkdir()
