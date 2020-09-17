from DV import DV
from V_time import V_Time
import time

class DV_Time(DV, V_Time):

    _name = "time_DV"
    _UID = ""
    _variable_label = "Time"
    _units = "s"
    _priority = 0
    _value_range = (0, 604800) # don't record more than a week
    _value = 0

    _is_covariate = True

    # Initializes reference time.
    # The reference time usually denotes the beginning of an experiment trial.
    def __init__(self, *args, **kwargs):
        print(self._variable_label)
        super(DV_Time, self).__init__(*args, **kwargs)
        print(self._variable_label)

    # Measure number of seconds relative to reference time
    def measure(self):

        value = (time.time() - self._t0)
        self.set_value(value)
