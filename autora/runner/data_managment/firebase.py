import json
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


def send_condition(collection_name: str, condition: Any, firebase_credentials: str):
    """
    Upload a condition to a firestore database

    Args:
        collection_name: the name of the study as given in firebase
        condition: the condition
        firebase_credentials: dict with the credentials for firebase
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    seq_col = db.collection(f"{collection_name}")
    doc_ref = seq_col.document("data")
    doc_ref.set({"data": None})
    doc_ref = seq_col.document("condition")
    condition_json = json.dumps(condition.tolist())
    doc_ref.set({"condition": condition_json})


def get_dependent_variable(collection_name: str, firebase_credentials: dict) -> Any:
    """
    get dependent variable from firestore database

    Args:
        collection_name: name of the collection as given in firebase
        firebase_credentials: credentials for firebase

    Returns:
        the dependent variable
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")
    doc_ref = seq_col.document("data")
    data = doc_ref.get().to_dict()["data"]
    return data
