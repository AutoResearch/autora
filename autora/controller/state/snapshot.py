""" Classes for storing and passing a cycle's state as an immutable snapshot. """
from dataclasses import dataclass, field
from typing import Dict, List

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection


@dataclass(frozen=True)
class ControllerState:
    """An object passed between and updated by processing steps in the Controller."""

    # Single values
    metadata: VariableCollection = field(default_factory=VariableCollection)
    params: Dict = field(default_factory=dict)

    # Sequences
    conditions: List[ArrayLike] = field(default_factory=list)
    observations: List[ArrayLike] = field(default_factory=list)
    theories: List[BaseEstimator] = field(default_factory=list)

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
        >>> s0 = ControllerState()
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ControllerState(metadata=VariableCollection(...), params={}, conditions=[],
                        observations=[], theories=[])

        We can update the params using the `.update` method:
        >>> s0.update(params={'first': 'params'})  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ControllerState(..., params={'first': 'params'}, ...)

        ... but the original object is unchanged:
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ControllerState(..., params={}, ...)

        For params, only one object is returned from the respective property:
        >>> s0.update(params={'first': 'params'}).update(params={'second': 'params'}).params
        {'second': 'params'}

        ... and the same applies to metadata:
        >>> from autora.variable import VariableCollection
        >>> (s0.update(metadata=VariableCollection(["1st IV"]))
        ...    .update(metadata=VariableCollection(["2nd IV"]))).metadata
        VariableCollection(independent_variables=['2nd IV'], dependent_variables=[], covariates=[])

        When we update the conditions, observations or theories, the respective list is extended:
        >>> s3 = s0.update(theories=["1st theory"])
        >>> s3
        ControllerState(..., theories=['1st theory'])

        ... so we can see the history of all the theories, for instance.
        >>> s3.update(theories=["2nd theory"])
        ControllerState(..., theories=['1st theory', '2nd theory'])

        The same applies to observations:
        >>> s4 = s0.update(observations=["1st observation"])
        >>> s4
        ControllerState(..., observations=['1st observation'], ...)

        >>> s4.update(observations=["2nd observation"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ControllerState(..., observations=['1st observation', '2nd observation'], ...)


        The same applies to conditions:
        >>> s5 = s0.update(conditions=["1st condition"])
        >>> s5
        ControllerState(..., conditions=['1st condition'], ...)

        >>> s5.update(conditions=["2nd condition"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ControllerState(..., conditions=['1st condition', '2nd condition'], ...)

        You can also update with multiple conditions, observations and theories:
        >>> s0.update(conditions=['c1', 'c2'])
        ControllerState(..., conditions=['c1', 'c2'], ...)

        >>> s0.update(theories=['t1', 't2'], metadata={'m': 1})
        ControllerState(metadata={'m': 1}, ..., theories=['t1', 't2'])

        >>> s0.update(theories=['t1'], observations=['o1'], metadata={'m': 1})
        ControllerState(metadata={'m': 1}, ..., observations=['o1'], theories=['t1'])


        Inputs to theories, observations and conditions must be Lists
        which can be cast to lists:
        >>> s0.update(theories='t1')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AssertionError: 't1' must be a list, e.g. `['t1']`?)

        """

        def _coalesce_lists(old, new):
            assert isinstance(
                old, List
            ), f"{repr(old)} must be a list, e.g. `[{repr(old)}]`?)"
            if new is not None:
                assert isinstance(
                    new, List
                ), f"{repr(new)} must be a list, e.g. `[{repr(new)}]`?)"
                return old + list(new)
            else:
                return old

        metadata_ = metadata or self.metadata
        params_ = params or self.params
        conditions_ = _coalesce_lists(self.conditions, conditions)
        observations_ = _coalesce_lists(self.observations, observations)
        theories_ = _coalesce_lists(self.theories, theories)
        return ControllerState(
            metadata_, params_, conditions_, observations_, theories_
        )
