from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


@dataclass(frozen=True)
class SimpleCycleData:
    """An object passed between and updated by processing steps in the SimpleCycle."""

    # Static
    metadata: VariableCollection

    # Aggregates each cycle from the:
    # ... Experimentalist
    conditions: List[np.ndarray]
    # ... Experiment Runner
    observations: List[np.ndarray]
    # ... Theorist
    theories: List[BaseEstimator]
