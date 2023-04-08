from time import sleep

from autora.runner.data_managment.firebase import get_dependent_variable, send_condition
from autora.runner.participant_managment.prolific import increase_participant_count
from autora.runner.runner import runner


def firebase_set(condition, **kwargs):
    """
    Set up the experiment on firebase (and prolific)
    Args:
        condition: the condition
        **kwargs:
    """
    send_condition(kwargs["collection_name"], condition, kwargs["firebase_credentials"])
    if "prolific_token" in kwargs.keys():
        increase_participant_count(kwargs["study_name"], kwargs["prolific_token"])


def firebase_get(**kwargs):
    """
    Get the dependent variable from firebase when it is ready
    Args:
        **kwargs:

    Returns:
        the dependent variavle

    """
    data = get_dependent_variable(
        kwargs["collection_name"], kwargs["firebase_credentials"]
    )
    while data is None:
        sleep(10)
        data = get_dependent_variable(
            kwargs["collection_name"], kwargs["firebase_credentials"]
        )
    return data


def firebase_prolific_runner(condition, **kwargs):
    """
    A runner that uses firebase to store the condition and the dependent variable and prolific to
    recruit participants.
    Args:
        condition: the condition
        **kwargs:

    Returns:
        the dependent variable

    """
    return runner(condition, firebase_set, firebase_get, **kwargs)
