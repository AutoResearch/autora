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

    def update(self, metadata=None, conditions=None, observations=None, theories=None):
        """
        Create a new object with updated values.

        The initial object is empty:
        >>> s0 = SimpleCycleData(None, [], [], [])
        >>> s0
        SimpleCycleData(metadata=None, conditions=[], observations=[], theories=[])

        We can update the metadata using the `.update` method:
        >>> from autora.variable import VariableCollection
        >>> s1 = s0.update(metadata=VariableCollection())
        >>> s1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(metadata=VariableCollection(...), conditions=[], observations=[],
                        theories=[])

        ... the original object is unchanged:
        >>> s0
        SimpleCycleData(metadata=None, conditions=[], observations=[], theories=[])

        We can update the metadata again:
        >>> s2 = s1.update(metadata=VariableCollection(["some IV"]))
        >>> s2  # doctest: +ELLIPSIS
        SimpleCycleData(metadata=VariableCollection(independent_variables=['some IV'],...), ...)

        ... and we see that there is only ever one metadata object returned.

        When we update the conditions, observations or theories, the respective list is extended:
        >>> s3 = s0.update(theories=["1st theory"])
        >>> s3
        SimpleCycleData(..., theories=['1st theory'])

        ... so we can see the history of all the theories, for instance.
        >>> s3.update(theories=["2nd theory"])
        SimpleCycleData(..., theories=['1st theory', '2nd theory'])

        The same for the observations:
        >>> s4 = s0.update(observations=["1st observation"])
        >>> s4
        SimpleCycleData(metadata=None, conditions=[], observations=['1st observation'], theories=[])

        >>> s4.update(observations=["2nd observation"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(metadata=None, conditions=[],
                        observations=['1st observation', '2nd observation'], theories=[])


        The same for the conditions:
        >>> s5 = s0.update(conditions=["1st condition"])
        >>> s5
        SimpleCycleData(metadata=None, conditions=['1st condition'], observations=[], theories=[])

        >>> s5.update(conditions=["2nd condition"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(metadata=None, conditions=['1st condition', '2nd condition'],
                        observations=[], theories=[])

        You can also update with multiple conditions, observations and theories:
        >>> s0.update(conditions=['c1', 'c2'])
        SimpleCycleData(metadata=None, conditions=['c1', 'c2'], observations=[], theories=[])

        >>> s0.update(theories=['t1', 't2'], metadata={'m': 1})
        SimpleCycleData(metadata={'m': 1}, conditions=[], observations=[], theories=['t1', 't2'])

        >>> s0.update(theories=['t1'], observations=['o1'], metadata={'m': 1})
        SimpleCycleData(metadata={'m': 1}, conditions=[], observations=['o1'], theories=['t1'])

        """

        def _coalesce_lists(old, new):
            if new is not None:
                return old + list(new)
            else:
                return old

        metadata_ = metadata or self.metadata
        conditions_ = _coalesce_lists(self.conditions, conditions)
        observations_ = _coalesce_lists(self.observations, observations)
        theories_ = _coalesce_lists(self.theories, theories)
        return SimpleCycleData(metadata_, conditions_, observations_, theories_)
