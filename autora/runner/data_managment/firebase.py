import time
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


def _sequence_to_db_object(iterable):
    """
    Convert a array into a dictionary for a database
    Args:
        iterable: an iterable

    Returns:
        a dict with keys 0, 1, 2..
    Examples:
        A simple range object can be converted into an array of dimension 2:
        >>> _sequence_to_db_object(range(3))
        {0: 0, 1: 1, 2: 2}

        A np.array with more two dimentions
        >>> import numpy as np
        >>> _sequence_to_db_object(np.array([[1, 2], [3, 4], [5, 6]]))
        {0: array([1, 2]), 1: array([3, 4]), 2: array([5, 6])}

        Not iterable
        >>> _sequence_to_db_object(3)
        {0: 3}
    """
    if not hasattr(iterable, "__iter__"):
        return {0: iterable}
    return {i: t for i, t in enumerate(iterable)}


def send_conditions(collection_name: str, conditions: Any, firebase_credentials: str):
    """
    Upload a condition to a firestore database

    Args:
        collection_name: the name of the study as given in firebase
        conditions: the condition to run
        firebase_credentials: dict with the credentials for firebase
    """

    # get the conditions with their indexes
    condition_dict = _sequence_to_db_object(conditions)

    # get the firebase collection (name of the study most probably)
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")

    # get the documents
    doc_ref_meta = seq_col.document("autora_meta")
    doc_ref_out = seq_col.document("autora_out")
    doc_ref_in = seq_col.document("autora_in")

    # set metadata
    # start_time and is_finished for each condition
    meta_dict = _sequence_to_db_object(
        [{"start_time": None, "finished": False}] * len(condition_dict)
    )
    meta_dict = {str(key): value for key, value in meta_dict.items()}
    doc_ref_meta.set(meta_dict)

    # reset the data
    col_ref = doc_ref_out.collection("observations")
    docs = col_ref.stream()
    for doc in docs:
        doc.reference.delete()

    col_ref = doc_ref_in.collection("conditions")
    docs = col_ref.stream()
    for doc in docs:
        doc.reference.delete()
    # setup db for conditions and observations
    for key in condition_dict:
        doc_ref_in.collection("conditions").document(str(key)).set(
            {str(key): condition_dict[key]}
        )
        doc_ref_out.collection("observations").document(str(key)).set({str(key): None})


def get_observations(collection_name: str, firebase_credentials: dict) -> Any:
    """
    get observations from firestore database

    Args:
        collection_name: name of the collection as given in firebase
        firebase_credentials: credentials for firebase

    Returns:
        observations
    """

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")

    doc_ref_out = seq_col.document("autora_out")

    col_ref = doc_ref_out.collection("observations")
    docs = col_ref.stream()
    observations = {}
    for doc in docs:
        observations.update(doc.reference.get().to_dict())
    return observations


def check_firebase_status(
    collection_name: str, firebase_credentials: dict, time_out: int
) -> str:
    """
    check the status of the condition

    Args:
        collection_name: name of the collection as given in firebase
        firebase_credentials: credentials for firebase
        time_out: time out for participants that started the condition
            but didn't finish (after this time spots are freed)

    Returns:
        Can have three different outcomes:
            (1) available -> no action needed, recruitment should be started (if paused)
            (2) finished -> collection of observations is finished
            (3) unavailable -> all conditions are running, recruitment should be paused
    """

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")

    doc_ref_meta = seq_col.document("autora_meta")
    meta_data = doc_ref_meta.get().to_dict()

    finished = True
    for key, value in meta_data.items():
        # return available if there are conditions that haven't been started
        if value["start_time"] is None:
            return "available"
        else:
            if not value["finished"]:
                unix_time_seconds = int(time.time())
                time_from_started = unix_time_seconds - value["start_time"]
                # check weather the started condition has timed out, if so, reset start_time and
                # return available
                if time_from_started > time_out:
                    doc_ref_meta.update({key: {"start_time": None, "finished": False}})
                    return "available"
                else:
                    finished = False
    if finished:
        # if all start_times are set and have data, condition is finished
        return "finished"
    # if all start_times are set, but there is no data for all of them, pause the condition
    return "unavailable"
