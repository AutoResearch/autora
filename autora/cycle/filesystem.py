import pickle


def save(data: object, file):
    """
    Serialize an arbitrary data object.

    Args:
        data: an arbitrary python object
        path: location to save to, by default a pickle file in the current working directory.

    Returns: None

    """
    pickle.dump(data, file)
    return


def load(file):
    data = pickle.load(file)
    return data
