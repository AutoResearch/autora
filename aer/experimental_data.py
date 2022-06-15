from dataclasses import dataclass

import pandas as pd
from variable import VariableCollection


@dataclass(frozen=True)
class ExperimentalData:
    """Immutable dataset."""

    metadata: VariableCollection
    data: pd.DataFrame


def load_experimental_data(
    filepath: str, metadata: VariableCollection
) -> ExperimentalData:
    data = pd.read_csv(filepath, header=0)
    for name in metadata.variable_names:
        assert name in data.columns
    e = ExperimentalData(metadata=metadata, data=data)
    return e
