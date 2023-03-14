from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


@dataclass(frozen=True)
class SimpleCycleData:
    """An object passed between and updated by processing steps in the SimpleCycle."""

    # Static
    metadata: VariableCollection = field(default_factory=VariableCollection)

    # Potentially variable parameters
    params: Dict = field(default_factory=dict)

    # Aggregates each cycle from the:
    # ... Experimentalist
    conditions: List[np.ndarray] = field(default_factory=list)
    # ... Experiment Runner
    observations: List[np.ndarray] = field(default_factory=list)
    # ... Theorist
    theories: List[BaseEstimator] = field(default_factory=list)
