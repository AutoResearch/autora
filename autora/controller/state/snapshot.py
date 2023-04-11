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
    parameters: Dict = field(default_factory=dict)

    # Sequences
    experiments: List[ArrayLike] = field(default_factory=list)
    observations: List[ArrayLike] = field(default_factory=list)
    models: List[BaseEstimator] = field(default_factory=list)

    def update(
        self,
        variables=None,
        parameters=None,
        experiments=None,
        observations=None,
        models=None,
    ):
        """
        Create a new object with updated values.

        The initial object is empty:
        >>> s0 = Snapshot()
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(variables=VariableCollection(...), parameters={}, experiments=[],
                        observations=[], models=[])

        We can update the parameters using the `.update` method:
        >>> s0.update(parameters={'first': 'parameters'}
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., parameters={'first': 'parameters'}, ...)

        ... but the original object is unchanged:
        >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Snapshot(..., parameters={}, ...)

        For parameters, only one object is returned from the respective property:
        >>> s0.update(parameters={'first': 'parameters'}
        ...  ).update(parameters={'second': 'parameters'}).parameters
        {'second': 'parameters'}

        ... and the same applies to variables:
        >>> from autora.variable import VariableCollection, IV
        >>> (s0.update(variables=VariableCollection([IV("1st IV")]))
        ...    .update(variables=VariableCollection([IV("2nd IV")]))).variables
        VariableCollection(independent_variables=[IV(name='2nd IV',...)], ...)

        When we update the experiments, observations or models, the respective list is extended:
        >>> s3 = s0.update(models=["1st model"])
        >>> s3
        Snapshot(..., models=['1st model'])

        ... so we can see the history of all the models, for instance.
        >>> s3.update(models=["2nd model"])
        Snapshot(..., models=['1st model', '2nd model'])

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

        You can also update with multiple experiments, observations and models:
        >>> s0.update(experiments=['c1', 'c2'])
        Snapshot(..., experiments=['c1', 'c2'], ...)

        >>> s0.update(models=['t1', 't2'], variables={'m': 1})
        Snapshot(variables={'m': 1}, ..., models=['t1', 't2'])

        >>> s0.update(models=['t1'], observations=['o1'], variables={'m': 1})
        Snapshot(variables={'m': 1}, ..., observations=['o1'], models=['t1'])


        Inputs to models, observations and experiments must be Lists
        which can be cast to lists:
        >>> s0.update(models='t1')  # doctest: +ELLIPSIS
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
        parameters_ = parameters or self.parameters
        experiments_ = _coalesce_lists(self.experiments, experiments)
        observations_ = _coalesce_lists(self.observations, observations)
        models_ = _coalesce_lists(self.models, models)
        return Snapshot(variables_, parameters_, experiments_, observations_, models_)
