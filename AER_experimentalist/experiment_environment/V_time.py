import time

class V_Time():

    _t0 = 0

    def __init__(self):
        self._t0 = time.time()

    # Resets reference time.
    def reset(self):
        self._t0 = time.time()