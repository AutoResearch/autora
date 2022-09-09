import time

from autora.variable import DV, IV


class VTime:
    """
    A class representing time as a general experimental variable.
    """

    _t0 = 0

    def __init__(self):
        """
        Initializes the time.
        """
        self._t0 = time.time()

    # Resets reference time.
    def reset(self):
        """
        Resets the time.
        """
        self._t0 = time.time()


class IVTime(IV, VTime):
    """
    A class representing time as an independent variable.
    """

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
        """
        Initializes the time as independent variable.

        For arguments, see [autora.variable.Variable][autora.variable.Variable.__init__]
        """
        super(IVTime, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        """
        Waits for the specified time to pass.
        """

        t_wait = self.get_value() - (time.time() - self._t0)
        if t_wait <= 0:
            return
        else:
            time.sleep(t_wait)

    def disconnect(self):
        """
        Disconnects the time.
        """
        pass


class DVTime(DV, VTime):
    """
    A class representing time as a dependent variable.
    """

    _name = "time_DV"
    _UID = ""
    _variable_label = "Time"
    _units = "s"
    _priority = 0
    _value_range = (0, 604800)  # don't record more than a week
    _value = 0

    _is_covariate = True

    # Initializes reference time.
    # The reference time usually denotes the beginning of an experiment trial.
    def __init__(self, *args, **kwargs):
        """
        Initializes the time as dependent variable. The reference time usually denotes
        the beginning of an experiment trial.

        For arguments, see [autora.variable.Variable][autora.variable.Variable.__init__]
        """
        print(self._variable_label)
        super(DVTime, self).__init__(*args, **kwargs)
        print(self._variable_label)

    # Measure number of seconds relative to reference time
    def measure(self):
        """
        Measures the time in seconds relative to the reference time.
        """
        value = time.time() - self._t0
        self.set_value(value)
