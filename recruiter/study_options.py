from enum import Enum


class StudyAction(str, Enum):
    """
    Used to edit published studies.
    Easily stores the options in case of API changes in the future.
    """

    PUBLISH = "PUBLISH"
    PAUSE = "PAUSE"
    START = "START"
    # can only continue after increasing study participant limit
    STOP = "STOP"


class ProlificIdOptions(str, Enum):
    """
    Used to determine the participant ID collection
    method during study drafting.
    """

    # collected by question we ask on our website
    QUESTION = "question"
    # use prolific integration to get ID from survey tool
    URL_PARAMS = "url_parameters"
    # elect not to record IDs
    NO_REQ = "not_required"


class CompletionOptions(str, Enum):
    URL = "url"
    CODE = "code"


class DeviceOptions(str, Enum):
    """
    Allows us to elect devices the participants can use.
    """

    DESKTOP = "desktop"
    TABLET = "tablet"
    MOBILE = "mobile"


class PeripheralOptions(str, Enum):
    """
    Specifies additional requirements.
    """

    AUDIO = "audio"
    CAM = "camera"
    DL = "download"
    MIC = "microphone"
