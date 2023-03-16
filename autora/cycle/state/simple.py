from dataclasses import dataclass
from typing import Dict, List

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


@dataclass(frozen=True)
class SimpleCycleData:
    """An object passed between and updated by processing steps in the SimpleCycle."""

    # Single values
    metadata: VariableCollection
    params: Dict

    # Sequences
    conditions: List[ArrayLike]
    observations: List[ArrayLike]
    theories: List[BaseEstimator]

    def update(
        self,
        metadata=None,
        params=None,
        conditions=None,
        observations=None,
        theories=None,
    ):
        """
        Create a new object with updated values.

        The initial object is empty:
        >>> s0 = SimpleCycleData(None, {}, [], [], [])
        >>> s0
        SimpleCycleData(metadata=None, params={}, conditions=[], observations=[], theories=[])

        We can update the metadata using the `.update` method:
        >>> from autora.variable import VariableCollection
        >>> s1 = s0.update(metadata=VariableCollection())
        >>> s1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(metadata=VariableCollection(...), params={}, conditions=[], observations=[],
                        theories=[])

        ... the original object is unchanged:
        >>> s0
        SimpleCycleData(metadata=None, params={}, conditions=[], observations=[], theories=[])

        We can update the metadata again:
        >>> s2 = s1.update(metadata=VariableCollection(["some IV"]))
        >>> s2  # doctest: +ELLIPSIS
        SimpleCycleData(metadata=VariableCollection(independent_variables=['some IV'],...), ...)

        ... and we see that there is only ever one metadata object returned.

        Params is treated the same way as metadata:
        >>> s0.update(params={'first': 'params'})
        SimpleCycleData(..., params={'first': 'params'}, ...)

        >>> s0.update(params={'first': 'params'}).update(params={'second': 'params'}).params
        {'second': 'params'}

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
        SimpleCycleData(..., observations=['1st observation'], ...)

        >>> s4.update(observations=["2nd observation"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(..., observations=['1st observation', '2nd observation'], ...)


        The same for the conditions:
        >>> s5 = s0.update(conditions=["1st condition"])
        >>> s5
        SimpleCycleData(..., conditions=['1st condition'], ...)

        >>> s5.update(conditions=["2nd condition"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        SimpleCycleData(..., conditions=['1st condition', '2nd condition'], ...)

        You can also update with multiple conditions, observations and theories:
        >>> s0.update(conditions=['c1', 'c2'])
        SimpleCycleData(..., conditions=['c1', 'c2'], ...)

        >>> s0.update(theories=['t1', 't2'], metadata={'m': 1})
        SimpleCycleData(metadata={'m': 1}, ..., theories=['t1', 't2'])

        >>> s0.update(theories=['t1'], observations=['o1'], metadata={'m': 1})
        SimpleCycleData(metadata={'m': 1}, ..., observations=['o1'], theories=['t1'])

        """

        def _coalesce_lists(old, new):
            if new is not None:
                return old + list(new)
            else:
                return old

        metadata_ = metadata or self.metadata
        params_ = params or self.params
        conditions_ = _coalesce_lists(self.conditions, conditions)
        observations_ = _coalesce_lists(self.observations, observations)
        theories_ = _coalesce_lists(self.theories, theories)
        return SimpleCycleData(
            metadata_, params_, conditions_, observations_, theories_
        )
