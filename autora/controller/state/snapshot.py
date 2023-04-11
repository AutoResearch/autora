""" Classes for storing and passing a cycle's state as an immutable snapshot. """
from dataclasses import dataclass, field
from typing import Dict, List

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.controller.protocol import SupportsControllerStateFields
from autora.variable import VariableCollection


@dataclass(frozen=True)
class Snapshot(SupportsControllerStateFields):
    """An object passed between and updated by processing steps in the Controller."""

    # Single values
    variables: VariableCollection = field(default_factory=VariableCollection)
    params: Dict = field(default_factory=dict)

    # Sequences
    experiments: List[ArrayLike] = field(default_factory=list)
    observations: List[ArrayLike] = field(default_factory=list)
    theories: List[BaseEstimator] = field(default_factory=list)

    def update(
        self,
        variables=None,
        params=None,
        experiments=None,
        observations=None,
        theories=None,
    ):
        """
        Create a new object with updated values.

        The initial object is empty:
        >>> s0 = Snapshot()
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(variables=VariableCollection(...), params={}, experiments=[],
                        observations=[], theories=[])

        We can update the params using the `.update` method:
        >>> s0.update(params={'first': 'params'})  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., params={'first': 'params'}, ...)

        ... but the original object is unchanged:
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., params={}, ...)

        For params, only one object is returned from the respective property:
        >>> s0.update(params={'first': 'params'}).update(params={'second': 'params'}).params
        {'second': 'params'}

        ... and the same applies to variables:
        >>> from autora.variable import VariableCollection, IV
        >>> (s0.update(variables=VariableCollection([IV("1st IV")]))
        ...    .update(variables=VariableCollection([IV("2nd IV")]))).variables
        VariableCollection(independent_variables=[IV(name='2nd IV',...)], ...)

        When we update the experiments, observations or theories, the respective list is extended:
        >>> s3 = s0.update(theories=["1st theory"])
        >>> s3
        Snapshot(..., theories=['1st theory'])

        ... so we can see the history of all the theories, for instance.
        >>> s3.update(theories=["2nd theory"])
        Snapshot(..., theories=['1st theory', '2nd theory'])

        The same applies to observations:
        >>> s4 = s0.update(observations=["1st observation"])
        >>> s4
        Snapshot(..., observations=['1st observation'], ...)

        >>> s4.update(observations=["2nd observation"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., observations=['1st observation', '2nd observation'], ...)


        The same applies to experiments:
        >>> s5 = s0.update(experiments=["1st experiment"])
        >>> s5
        Snapshot(..., experiments=['1st experiment'], ...)

        >>> s5.update(experiments=["2nd experiment"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., experiments=['1st experiment', '2nd experiment'], ...)

        You can also update with multiple experiments, observations and theories:
        >>> s0.update(experiments=['c1', 'c2'])
        Snapshot(..., experiments=['c1', 'c2'], ...)

        >>> s0.update(theories=['t1', 't2'], variables={'m': 1})
        Snapshot(variables={'m': 1}, ..., theories=['t1', 't2'])

        >>> s0.update(theories=['t1'], observations=['o1'], variables={'m': 1})
        Snapshot(variables={'m': 1}, ..., observations=['o1'], theories=['t1'])


        Inputs to theories, observations and experiments must be Lists
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

        variables_ = variables or self.variables
        params_ = params or self.params
        experiments_ = _coalesce_lists(self.experiments, experiments)
        observations_ = _coalesce_lists(self.observations, observations)
        theories_ = _coalesce_lists(self.theories, theories)
        return Snapshot(variables_, params_, experiments_, observations_, theories_)
