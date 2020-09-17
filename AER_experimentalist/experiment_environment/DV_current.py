from DV import DV
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_industrial_dual_0_20ma_v2 import BrickletIndustrialDual020mAV2

class DV_Current(DV):

    _name = "current0"
    _UID = "Hfg"
    _variable_label = "Current 0"
    _units = "mA"
    _priority = 0
    _value_range = (0, 2000)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223
    channel = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        super(DV_Current, self).__init__(*args, **kwargs)

        self._ipcon = IPConnection()  # Create IP connection
        self._id020 = BrickletIndustrialDual020mAV2(self._UID, self._ipcon)  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if (self._name == "current1"):
            self.channel = 1
        else:
            self.channel = 0

    # Clean up measurement device.
    def __clean_up__(self):

        self._ipcon.disconnect()

    # Waits until specified time has passed relative to reference time
    def measure(self):
        current = self._id020.get_current(self.channel)
        self.set_value(current/1000000.0)
