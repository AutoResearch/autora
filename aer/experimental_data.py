from dataclasses import dataclass

import pandas as pd
from variable import VariableCollection


@dataclass(frozen=True)
class ExperimentalData:
    """Immutable dataset."""

    metadata: VariableCollection
    data: pd.DataFrame

    @property
    def dataframe(self) -> pd.DataFrame:
        """The data as a pandas DataFrame."""
        return self.data


def load_experimental_data(
    filepath: str, metadata: VariableCollection
) -> ExperimentalData:
    data = pd.read_csv(filepath, header=0)
    for name in metadata.variable_names:
        assert name in data.columns
    e = ExperimentalData(metadata=metadata, data=data)
    return e


def combine_experimental_data(
    d1: ExperimentalData, d2: ExperimentalData, *dn: ExperimentalData
) -> ExperimentalData:
    assert d1.metadata == d2.metadata, (
        f"Metadata must be identical but arent: " f"{d1.metadata} != {d2.metadata} "
    )

    metadata = d1.metadata
    data = pd.concat(d1.dataframe, d2.dataframe)

    e = ExperimentalData(metadata=metadata, data=data)

    if len(dn) > 0:
        return combine_experimental_data(e, dn[0], *dn[1:])

    return e
