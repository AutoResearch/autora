""" Contains AutoRA Cycle Planners, which interrogate the cycle state and then plan the next
step.  """

import random

from autora.cycle.protocol.v1 import Cycle
from autora.cycle.result import Result, ResultKind


def last_result_kind_planner(cycle: Cycle):
    """
    Chooses the operation based on the last result, e.g. new theory -> run experimentalist.

    Interpretation: The "traditional" AutoRA Cycle.
    """

    try:
        last_result = cycle.state[-1]
    except IndexError:
        last_result = Result(None, None)

    callback = {
        None: cycle.experimentalist,
        ResultKind.THEORY: cycle.experimentalist,
        ResultKind.CONDITION: cycle.experiment_runner,
        ResultKind.OBSERVATION: cycle.theorist,
    }[last_result.kind]

    return callback


def random_operation_planner(cycle: Cycle):
    """
    Chooses a random operation, ignoring any data which already exist.

    Interpretation: A mercurial research assistant who doesn't remember what they did last.
    """
    options = [
        cycle.experimentalist,
        cycle.experiment_runner,
        cycle.theorist,
    ]
    choice = random.choice(options)
    return choice
