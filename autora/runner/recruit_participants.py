import json

import requests

API_VERSION = "v1/"
BASE_URL = f"https://api.prolific.co/api/{API_VERSION}"
token = json.load(open("prolific.json"))["token"]
HEADERS = {"Authorization": f"Token {token}"}


def _list_studies():
    """
    Returns list of all studies on Prolific account.
    """
    studies = requests.get(f"{BASE_URL}studies/", headers=HEADERS)
    return studies.json()


def _get_id_from_name(study_name: str):
    lst = _list_studies()["results"]
    for s in lst:
        if s["name"] == study_name:
            return s["id"]
    return ""


def _update_study(study_id: str, **kwargs) -> bool:
    """
    Updates the parameters of a given study.
    If a study is already published, only internal_name
    and total_available_places can be updated.
    """
    study = requests.patch(
        f"{BASE_URL}studies/{study_id}/", headers=HEADERS, json=kwargs
    )
    return study.status_code < 400


def _retrieve_study(study_id: str):
    """
    Retrieves information about study given its ID.
    """
    study = requests.get(f"{BASE_URL}studies/{study_id}/", headers=HEADERS)
    return study.json()


def increase_participant_count(study_name: str, increment: int):
    """
    Convenience method specifically meant for increasing participant count
    """
    study_id = _get_id_from_name(study_name)
    available_places = _retrieve_study(study_id)["total_available_places"]
    return _update_study(study_id, total_available_places=available_places + 1)


increase_participant_count("autora", 1)
