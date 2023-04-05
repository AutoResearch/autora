import firebase_admin
from firebase_admin import credentials, firestore


def download_data(study):
    """
    get dependent variable from firestore database

    Args:
        study: name of the study

    Returns:
        the dependent variable
    """
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{study}")
    doc_ref = seq_col.document("data")
    data = doc_ref.get().to_dict()["data"]
    return data


download_data("autora")
