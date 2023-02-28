""" Contains AutoRA Cycle Planners, which interrogate the cycle state and then plan the next
step.  """

import random

from autora.cycle.protocol import Cycle
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
        None: cycle.run_experimentalist,
        ResultKind.THEORY: cycle.run_experimentalist,
        ResultKind.CONDITION: cycle.run_experiment_runner,
        ResultKind.OBSERVATION: cycle.run_theorist,
    }[last_result.kind]

    return callback


def random_operation_planner(cycle: Cycle):
    """
    Chooses a random operation, ignoring any data which already exist.

    Interpretation: A mercurial research assistant who doesn't remember what they did last.
    """
    options = [
        cycle.run_experimentalist,
        cycle.run_experiment_runner,
        cycle.run_theorist,
    ]
    choice = random.choice(options)
    return choice
