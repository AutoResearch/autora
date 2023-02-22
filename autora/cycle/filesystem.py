import pickle


class PickleDeSerializer:
    @staticmethod
    def save(data: object, file):
        """
        Serialize an arbitrary data object.

        Args:
            data: an arbitrary python object
            file: location to save to

        Returns: None

        """
        pickle.dump(data, file)
        return

    @staticmethod
    def load(file):
        data = pickle.load(file)
        return data
