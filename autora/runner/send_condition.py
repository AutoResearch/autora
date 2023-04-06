from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


def upload_condition(
    collection_name: str, condition: Any, path: str = "serviceAccountKey.json"
):
    """
    Upload a condition to a firestore database

    Args:
        collection_name: the name of the study as given in firebase
        condition: the condition
        path:  path to the file with the credentials for firebase
    """
    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")
    doc_ref = seq_col.document("condition")
    doc_ref.set({"condition": condition})


def reset_data(collection_name: str, path: str = "serviceAccountKey.json"):
    """
    Reset the dependent variable to none

    Args:
        collection_name: the name of the study as given in firebase
        path:  path to the file with the credentials for firebase

    Returns:
        the dependent variable
    """
    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")
    doc_ref = seq_col.document("data")
    doc_ref.set({"data": None})
