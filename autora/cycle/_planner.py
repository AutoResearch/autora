import random
from typing import Any, Protocol

from autora.cycle._executor import (
    Executor,
    SupportsExperimentalistExperimentRunnerTheorist,
    SupportsFullCycle,
)
from autora.cycle._state import ResultKind, SupportsConditionsObservationsTheories


class Planner(Protocol):
    def __call__(self, state, executor_collection) -> Executor:
        ...


def full_cycle_planner(state: Any, executor_collection: SupportsFullCycle):
    """Always returns the `full_cycle` method."""
    return executor_collection.full_cycle


def last_result_kind_planner(
    state: SupportsConditionsObservationsTheories,
    executor_collection: SupportsExperimentalistExperimentRunnerTheorist,
):
    """
    Chooses the operation based on the last result, e.g. new theory -> run experimentalist.

    Interpretation: The "traditional" AutoRA Controller – a systematic research assistant.

    Examples:
        We initialize a new CycleState to run our planner on:
        >>> from autora.cycle._state import CycleState
        >>> state = CycleState()

        We simulate a productive executor_collection using a SimpleNamespace
        >>> from types import SimpleNamespace
        >>> executor_collection = SimpleNamespace(
        ...     experimentalist = "experimentalist",
        ...     experiment_runner = "experiment_runner",
        ...     theorist = "theorist",
        ... )

        Based on the results available in the state, we can get the next kind of executor we need.
        When we have no results of any kind, we get an experimentalist:
        >>> last_result_kind_planner(state, executor_collection)
        'experimentalist'

        ... or if we had produced conditions, then we could run an experiment
        >>> state.update("some theory", kind="CONDITION")
        >>> last_result_kind_planner(state, executor_collection)
        'experiment_runner'

        ... or if we last produced observations, then we could now run the theorist:
        >>> state.update("some theory", kind="OBSERVATION")
        >>> last_result_kind_planner(state, executor_collection)
        'theorist'

        ... or if we last produced a theory, then we could now run the experimentalist:
        >>> state.update("some theory", kind="THEORY")
        >>> last_result_kind_planner(state, executor_collection)
        'experimentalist'

    """

    try:
        last_result_kind = state.conditions_observations_theories[-1].kind
    except IndexError:
        last_result_kind = None

    callback = {
        None: executor_collection.experimentalist,
        ResultKind.THEORY: executor_collection.experimentalist,
        ResultKind.CONDITION: executor_collection.experiment_runner,
        ResultKind.OBSERVATION: executor_collection.theorist,
    }[last_result_kind]

    return callback


def random_operation_planner(
    _, executor_collection: SupportsExperimentalistExperimentRunnerTheorist
):
    """
    Chooses a random operation, ignoring any data which already exist.

    Interpretation: A mercurial PI with good technique but poor planning, who doesn't remember what
    they did last.

    Examples:
        This planner completely ignores the state, so we can simulate it by using anything we
        like, here we choose the empty set:
        >>> state = {}

        We simulate a productive executor_collection using a SimpleNamespace
        >>> from types import SimpleNamespace
        >>> executor_collection = SimpleNamespace(
        ...     experimentalist = "experimentalist",
        ...     experiment_runner = "experiment_runner",
        ...     theorist = "theorist",
        ... )

        For reproducibility, we seed the random number generator consistently:
        >>> from random import seed
        >>> seed(42)

        Now we can begin to see which operations are returned by the planner. The first (for this
        seed) is the theorist.
        >>> random_operation_planner(state, executor_collection)
        'theorist'

        If we evaluate again, a random executor will be suggested each time
        >>> [random_operation_planner(state, executor_collection) for i in range(5)]
        ['experimentalist', 'experimentalist', 'theorist', 'experiment_runner', 'experimentalist']

    """
    options = [
        executor_collection.experimentalist,
        executor_collection.experiment_runner,
        executor_collection.theorist,
    ]
    choice = random.choice(options)
    return choice
