"""
Functions which look at state and output the next function to execute.
"""
import random

from autora.cycle.protocol.v1 import (
    ResultKind,
    SupportsCycleStateHistory,
    SupportsExperimentalistExperimentRunnerTheorist,
    SupportsFullCycle,
)


def full_cycle_planner(_, executor_collection: SupportsFullCycle):
    """Always returns the `full_cycle` method."""
    return executor_collection.full_cycle


def last_result_kind_planner(
    state: SupportsCycleStateHistory,
    executor_collection: SupportsExperimentalistExperimentRunnerTheorist,
):
    """
    Chooses the operation based on the last result, e.g. new theory -> run experimentalist.

    Interpretation: The "traditional" AutoRA Controller – a systematic research assistant.

    Examples:
        We initialize a new list to run our planner on:
        >>> from autora.cycle.state import CycleStateHistory
        >>> state_ = CycleStateHistory()

        We simulate a productive executor_collection using a SimpleNamespace
        >>> from types import SimpleNamespace
        >>> executor_collection_ = SimpleNamespace(
        ...     experimentalist = "experimentalist",
        ...     experiment_runner = "experiment_runner",
        ...     theorist = "theorist",
        ... )

        Based on the results available in the state, we can get the next kind of executor we need.
        When we have no results of any kind, we get an experimentalist:
        >>> last_result_kind_planner(state_, executor_collection_)
        'experimentalist'

        ... or if we had produced conditions, then we could run an experiment
        >>> from autora.cycle.state import Result
        >>> state_ = state_.update(conditions=["some condition"])
        >>> last_result_kind_planner(state_, executor_collection_)
        'experiment_runner'

        ... or if we last produced observations, then we could now run the theorist:
        >>> state_ = state_.update(observations=["some observation"])
        >>> last_result_kind_planner(state_, executor_collection_)
        'theorist'

        ... or if we last produced a theory, then we could now run the experimentalist:
        >>> state_ = state_.update(theories=["some theory"])
        >>> last_result_kind_planner(state_, executor_collection_)
        'experimentalist'

    """

    filtered_history = state.filter_by(
        kind={ResultKind.CONDITION, ResultKind.OBSERVATION, ResultKind.THEORY}
    ).history

    try:
        last_result_kind = filtered_history[-1].kind
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
        We simulate a productive executor_collection using a SimpleNamespace
        >>> from types import SimpleNamespace
        >>> executor_collection_ = SimpleNamespace(
        ...     experimentalist = "experimentalist",
        ...     experiment_runner = "experiment_runner",
        ...     theorist = "theorist",
        ... )

        For reproducibility, we seed the random number generator consistently:
        >>> from random import seed
        >>> seed(42)

        Now we can begin to see which operations are returned by the planner. The first (for this
        seed) is the theorist. (The first argument is provided for compatibility with the
        protocol, but is ignored.)
        >>> random_operation_planner([], executor_collection_)
        'theorist'

        If we evaluate again, a random executor will be suggested each time
        >>> [random_operation_planner([], executor_collection_) for i in range(5)]
        ['experimentalist', 'experimentalist', 'theorist', 'experiment_runner', 'experimentalist']

    """
    options = [
        executor_collection.experimentalist,
        executor_collection.experiment_runner,
        executor_collection.theorist,
    ]
    choice = random.choice(options)
    return choice
