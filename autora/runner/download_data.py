from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


def download_data(collection_name: str, path: str = "serviceAccountKey.json") -> Any:
    """
    get dependent variable from firestore database

    Args:
        collection_name: name of the collection as given in firebase
        path: path to the file with the credentials for firebase

    Returns:
        the dependent variable
    """
    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{collection_name}")
    doc_ref = seq_col.document("data")
    data = doc_ref.get().to_dict()["data"]
    return data
