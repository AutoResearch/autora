import time

from autora.runner.data_managment.firebase import (
    check_firebase_status,
    get_observations,
    send_conditions,
)
from autora.runner.participant_managment.prolific import (
    check_prolific_status,
    pause_study,
    setup_study,
    start_study,
)


def firebase_run(conditions, **kwargs):
    """
    Running an experiment with firebase to host the experiment and store the data.
    Args:
        conditions: the conditions
        **kwargs:  configuration of the experiment (this typically doesn't vary from cycle to cycle)

    Returns:
        observations
    """
    firebase_credentials = kwargs["firebase_credentials"]
    time_out = kwargs["time_out"]
    sleep_time = kwargs["sleep_time"]

    # set up study on firebase
    send_conditions("autora", conditions, firebase_credentials)

    # run the experiment as long as not all conditions are met
    while True:
        check_firebase = check_firebase_status("autora", firebase_credentials, time_out)
        if check_firebase == "finished":
            # get observations returns a dict
            observation = get_observations("autora", firebase_credentials)
            observation_list = [observation[key] for key in sorted(observation.keys())]
            return observation_list
        time.sleep(sleep_time)


def firebase_prolific_run(conditions, **kwargs):
    """
    Running an experiment with firebase to host the experiment and store the data
    and prolific to recruite participants
    Args:
        conditions: the conditions
        **kwargs: configuration of the experiment (this typically doesn't vary from cycle to cycle)

    Returns:
        observations
    """

    # set up study on firebase
    send_conditions("autora", conditions, kwargs["firebase_credentials"])

    # set up study on prolific
    prolific_dict = setup_study(
        kwargs["study_name"],
        kwargs["study_description"],
        kwargs["study_url"],
        kwargs["study_completion_time"],
        kwargs["prolific_token"],
    )

    # get the specification on prolific
    time_out = prolific_dict["maximum_allowed_time"]
    study_id = prolific_dict["id"]

    while True:
        # check firebase
        check_firebase = check_firebase_status(
            "autora", kwargs["firebase_credentials"], time_out
        )
        # check prolific
        if prolific_dict:
            check_prolific = check_prolific_status(study_id, kwargs["prolific_token"])
            if (
                check_prolific["number_of_submissions"]
                >= check_prolific["total_available_places"]
            ):
                return get_observations("autora", kwargs["firebase_credentials"])

        # firebase places available
        if check_firebase == "finished":
            return get_observations("autora", kwargs["firebase_credentials"])
        if check_firebase == "available":
            if check_prolific["status"] == "PAUSED":
                start_study(
                    study_name=study_id, prolific_token=kwargs["prolific_token"]
                )
        if check_prolific == "unavailable":
            if check_prolific["status"] == "STARTED":
                pause_study(
                    study_name=study_id, prolific_token=kwargs["prolific_token"]
                )
        time.sleep(kwargs["sleep_time"])


def firebase_prolific_runner(**kwargs):
    """
    A runner that uses firebase to store the condition and the dependent variable and prolific to
    recruit participants.
    Args:
        **kwargs: the configuration of the experiment

    Returns:
        the runner
    """

    def runner(x):
        return firebase_prolific_run(x, **kwargs)

    return runner


def firebase_runner(**kwargs):
    """
    A runner that uses firebase to store the condition and the dependent variable and prolific to
    recruit participants.
    Args:
        **kwargs: the configuration of the experiment

    Returns:
        the runner
    """

    def runner(x):
        return firebase_run(x, **kwargs)

    return runner
