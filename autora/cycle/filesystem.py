import json
import pickle


class PickleDeSerializer:
    @staticmethod
    def dump(data: object, file):
        """
        Serialize an arbitrary data object.

        Args:
            data: an arbitrary python object
            file: location to dump to

        Returns: None

        """
        pickle.dump(data, file)
        return

    @staticmethod
    def load(file):
        data = pickle.load(file)
        return data


class JSONDeSerializer:
    @staticmethod
    def dump(data: object, file):
        """
        Serialize an arbitrary data object.

        Args:
            data: an arbitrary python object
            file: location to dump to

        Returns: None

        """
        json.dump(data, file)
        return

    @staticmethod
    def load(file):
        data = json.load(file)
        return data
