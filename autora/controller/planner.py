"""
Functions which look at state and output the next function name to execute.
"""
import random

from autora.controller.protocol.v1 import ResultKind, SupportsControllerStateHistory


def full_cycle_planner(_):
    """Always returns the `full_cycle` method.

    Examples:
        The full_cycle_planner always returns the full cycle Executor
        >>> full_cycle_planner([])
        'full_cycle'

    """
    return "full_cycle"


def last_result_kind_planner(state: SupportsControllerStateHistory):
    """
    Chooses the operation based on the last result, e.g. new theory -> run experimentalist.

    Interpretation: The "traditional" AutoRA Controller – a systematic research assistant.

    Examples:
        We initialize a new list to run our planner on:
        >>> from autora.controller.state import History
        >>> state_ = History()

        Based on the results available in the state, we can get the next kind of executor we need.
        When we have no results of any kind, we get an experimentalist:
        >>> last_result_kind_planner(state_)
        'experimentalist'

        ... or if we had produced conditions, then we could run an experiment
        >>> state_ = state_.update(conditions=["some condition"])
        >>> last_result_kind_planner(state_)
        'experiment_runner'

        ... or if we last produced observations, then we could now run the theorist:
        >>> state_ = state_.update(observations=["some observation"])
        >>> last_result_kind_planner(state_)
        'theorist'

        ... or if we last produced a theory, then we could now run the experimentalist:
        >>> state_ = state_.update(theories=["some theory"])
        >>> last_result_kind_planner(state_)
        'experimentalist'

    """

    filtered_history = state.filter_by(
        kind={ResultKind.CONDITION, ResultKind.OBSERVATION, ResultKind.THEORY}
    ).history

    try:
        last_result_kind = filtered_history[-1].kind
    except IndexError:
        last_result_kind = None

    executor_name = {
        None: "experimentalist",
        ResultKind.THEORY: "experimentalist",
        ResultKind.CONDITION: "experiment_runner",
        ResultKind.OBSERVATION: "theorist",
    }[last_result_kind]

    return executor_name


def random_operation_planner(_):
    """
    Chooses a random operation, ignoring any data which already exist.

    Interpretation: A mercurial PI with good technique but poor planning, who doesn't remember what
    they did last.

    Examples:
        We simulate a productive executor_collection using a simple dict
        >>> executor_collection_ = dict(
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
        >>> random_operation_planner([])
        'theorist'

        If we evaluate again, a random executor will be suggested each time
        >>> [random_operation_planner([]) for i in range(5)]
        ['experimentalist', 'experimentalist', 'theorist', 'experiment_runner', 'experimentalist']

    """
    choice = random.choice(["experimentalist", "experiment_runner", "theorist"])
    return choice
