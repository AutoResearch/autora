import json

import requests


def _list_studies(path: str):
    """
    Returns list of all studies on Prolific account.
    """
    prolific_token = json.load(open(path))["token"]
    studies = requests.get(
        "https://api.prolific.co/api/v1/studies/",
        headers={"Authorization": f"Token {prolific_token}"},
    )
    return studies.json()


def _get_id_from_name(study_name: str, path: str):
    """
    Returns the id of a study given its name.
    """
    lst = _list_studies(path)["results"]
    for s in lst:
        if s["name"] == study_name:
            return s["id"]
    return ""


def _update_study(study_id: str, path: str, **kwargs) -> bool:
    """
    Updates the parameters of a given study.
    If a study is already published, only internal_name
    and total_available_places can be updated.
    """
    prolific_token = json.load(open(path))["token"]
    study = requests.patch(
        f"https://api.prolific.co/api/v1/studies/{study_id}/",
        headers={"Authorization": f"Token {prolific_token}"},
        json=kwargs,
    )
    return study.status_code < 400


def _retrieve_study(study_id: str, path: str):
    """
    Retrieves information about study given its ID.
    """
    prolific_token = json.load(open(path))["token"]
    study = requests.get(
        f"https://api.prolific.co/api/v1/studies/{study_id}/",
        headers={"Authorization": f"Token {prolific_token}"},
    )
    return study.json()


def increase_participant_count(
    study_name: str, increment: int = 1, path: str = "prolific.json"
) -> bool:
    """
    Increase participants on prolific to collect data for a new cycle

    Args:
        study_name: name of the study as given in prolific
        increment: number of participants to recruit for this cycle
        path: path to the prolific api token
    Returns:
        Returns True if participants got increased
    """
    study_id = _get_id_from_name(study_name, path)
    available_places = _retrieve_study(study_id, path)["total_available_places"]
    return _update_study(
        study_id, path, total_available_places=available_places + increment
    )
