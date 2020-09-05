from IV import IV
import time

class IV_Trial(IV):

    _variable_label = "trial"
    _UID = ""
    _name = "Trial"
    _units = "trials"
    _priority = 0
    _value_range = (0, 10000000)
    _value = 0

    def __init__(self, *args, **kwargs):
        super(IV_Trial, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        pass

    def __clean_up__(self):
        pass