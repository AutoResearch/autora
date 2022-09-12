from DV import DV
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_industrial_dual_analog_in_v2 import BrickletIndustrialDualAnalogInV2

class DV_Voltage(DV):

    _name = "voltage0"
    _UID = "MjY"
    _variable_label = "Voltage 0"
    _units = "mV"
    _priority = 0
    _value_range = (-3500, 3500)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223

    channel = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        super(DV_Voltage, self).__init__(*args, **kwargs)

        self._ipcon = IPConnection()  # Create IP connection
        self._idai = BrickletIndustrialDualAnalogInV2(self._UID, self._ipcon)  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if(self._name == "voltage1"):
            self.channel = 1
        else:
            self.channel = 0

    # Clean up measurement device.
    def __clean_up__(self):
        self._ipcon.disconnect()


    # Waits until specified time has passed relative to reference time
    def measure(self):
        value = self._idai.get_voltage(self.channel)
        self.set_value(value)
