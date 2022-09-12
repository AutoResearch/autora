from IV import IV
from V_time import V_Time
import time

class IV_Time(IV, V_Time):

    _name = "time_IV"
    _UID = ""
    _variable_label = "Time"
    _units = "s"
    _priority = 0
    _value_range = (0, 3600)
    _value = 0

    # Initializes reference time.
    # The reference time usually denotes the beginning of an experiment trial.
    def __init__(self, *args, **kwargs):
        super(IV_Time, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):

        t_wait = self.get_value() - (time.time() - self._t0)
        if t_wait <= 0:
            return
        else:
            time.sleep(t_wait)

    def __clean_up__(self):
        pass