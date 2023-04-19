import random
import string
from typing import Any, List

import requests

from autora.runner.participant_managment.eligibilityOptions import EligibilityOptions


def _list_studies(prolific_token: str):
    """
    Returns list of all studies on Prolific account.
    """
    studies = requests.get(
        "https://api.prolific.co/api/v1/studies/",
        headers={"Authorization": f"Token {prolific_token}"},
    )
    return studies.json()


def _get_id_from_name(study_name: str, prolific_token: str):
    """
    Returns the id of a study given its name.
    """
    lst = _list_studies(prolific_token)["results"]
    for s in lst:
        if s["name"] == study_name:
            return s["id"]
    return ""


def _update_study(study_id: str, prolific_token: str, **kwargs) -> bool:
    """
    Updates the parameters of a given study.
    If a study is already published, only internal_name
    and total_available_places can be updated.
    """
    study = requests.patch(
        f"https://api.prolific.co/api/v1/studies/{study_id}/",
        headers={"Authorization": f"Token {prolific_token}"},
        json=kwargs,
    )
    return study.status_code < 400


def _retrieve_study(study_id: str, prolific_token: str):
    """
    Retrieves information about study given its ID.
    """
    study = requests.get(
        f"https://api.prolific.co/api/v1/studies/{study_id}/",
        headers={"Authorization": f"Token {prolific_token}"},
    )
    return study.json()


def check_prolific_status(study_id: str, prolific_token: str) -> dict:
    """
    Check the status of a study
    Args:
        study_id: id of the study
        prolific_token: a prolific api token

    Returns:
        Status of the study (Paused, Started, Finished ...)
    """
    study = _retrieve_study(study_id, prolific_token)
    keys_to_include = [
        "total_available_places",
        "places_taken",
        "status",
        "number_of_submissions",
    ]
    return dict((key, value) for key, value in study.items() if key in keys_to_include)


def increase_participant_count(
    study_name: str, prolific_token: str, increment: int = 1
) -> bool:
    """
    Increase participants on prolific to collect data for a new cycle

    Args:
        study_name: name of the study as given in prolific
        increment: number of participants to recruit for this cycle
        prolific_token: a prolific api token
    Returns:
        Returns True if participants got increased
    """
    study_id = _get_id_from_name(study_name, prolific_token)
    available_places = _retrieve_study(study_id, prolific_token)[
        "total_available_places"
    ]
    return _update_study(
        study_id, prolific_token, total_available_places=available_places + increment
    )


def setup_study(
    name: str,
    description: str,
    external_study_url: str,
    estimated_completion_time: int,
    exclude_studies: List[str] = ["default"],
    reward: int = 0,
    prolific_id_option: str = "url_parameters",
    completion_code: str = "",
    completion_option: str = "url",
    total_available_places: int = 1,
    eligibility_requirements: List[str] = ["default"],
    device_compatibility: List[str] = ["desktop"],
    peripheral_requirements=None,
    prolific_token: str = "",
) -> Any:
    """
    Allows for a study to be drafted given the following parameters.

    Args:
        name (str): Name that will be displayed on prolific
        description (str): Description of study for participants
        external_study_url (str): URL to experiment website
        estimated_completion_time (int): How long the study takes
        exclude_studies (list): Exclude participants that participated in previous studies
            (default is studies with the same name)
        prolific_id_option (ProlificIdOptions): Method of collecting subject ID
        completion_code (str): Code subject uses to mark experiment completion
        completion_option (CompletionOptions): Method of signifying participation
        total_available_places (int): Participant limit
        reward (int): Amount of payment for completion
        eligibility_requirements (list, optional): Allows various options to filter participants.
            Defaults to [] (no requirements).
        device_compatibility (list[DeviceOptions], optional): Allows selecting required devices.
            Defaults to [] (any device).
        peripheral_requirements (list[PeripheralOptions], optional):
            Allows specifying additional requirements. Defaults to [] (no other requirements).

    Returns:
        bool: Whether the request was successful or not
    """
    if eligibility_requirements is None:
        eligibility_requirements = []
    if exclude_studies is None:
        exclude_studies = []
    if exclude_studies == ["default"]:
        exclude_studies = [name]
    if eligibility_requirements == ["default"]:
        age_eligibility = EligibilityOptions.age(18, 55)
        nationality_eligibility = EligibilityOptions.nationality("United States", 1)
        vision_eligibility = EligibilityOptions.vision()
        language_eligibility = EligibilityOptions.first_language("English")
        eligibility_requirements = [
            age_eligibility,
            nationality_eligibility,
            vision_eligibility,
            language_eligibility,
        ]
    previous_studies = _list_studies(prolific_token)["results"]
    excludes = [
        {"name": s["name"], "id": s["id"]}
        for s in previous_studies
        if s["name"] in exclude_studies
    ]
    if excludes is not []:
        eligibility_requirements += [EligibilityOptions.previous_studies(excludes)]
    if device_compatibility is None:
        device_compatibility = []
    if peripheral_requirements is None:
        peripheral_requirements = []
    if completion_code == "":
        completion_code = "".join(
            random.choices(string.ascii_letters + string.digits, k=6)
        )
    if reward == 0:
        reward = round(20 * estimated_completion_time)  # 12$ per hour / 20¢ per minute

    # packages function parameters into dictionary
    data = locals()

    data["status"] = "PAUSED"

    study = requests.post(
        "https://api.prolific.co/api/v1/studies/",
        headers={"Authorization": f"Token {prolific_token}"},
        json=data,
    )

    # handles request failure
    if study.status_code >= 400:
        print(study.json())
        return False
    print(study.json())
    keys_to_include = ["id", "maximum_allowed_time"]
    return dict(
        (key, value) for key, value in study.json().items() if key in keys_to_include
    )


def _update_study_status(study_id: str, action: str, prolific_token: str):
    """
    Performs action on specified study. Default action is to publish
    the study.
    """
    data = {"action": action}
    study = requests.post(
        f"https://api.prolific.co/api/v1/studies//{study_id}/transition/",
        headers={"Authorization": f"Token {prolific_token}"},
        json=data,
    )
    if study.status_code != 400:
        return False
    return True


# number_of_submissions
def pause_study(study_id: str, prolific_token: str):
    """
    Pauses the study
    """
    return _update_study_status(study_id, "PAUSE", prolific_token)


def start_study(study_id: str, prolific_token: str):
    """
    Starts/Resumes the study
    """
    return _update_study_status(study_id, "START", prolific_token)
