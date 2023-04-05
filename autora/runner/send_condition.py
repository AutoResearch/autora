import firebase_admin
from firebase_admin import credentials, firestore


def upload_condition(study, condition):
    """
    Upload a condition to a firestore database

    Args:
        study: the name of the study
        condition: the condition
    """
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    seq_col = db.collection(f"{study}")
    doc_ref = seq_col.document("condition")
    doc_ref.set({"condition": condition})


upload_condition("autora", 0.1)
