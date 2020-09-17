from IV import IV
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_industrial_analog_out_v2 import BrickletIndustrialAnalogOutV2

class IV_Current(IV):

    _name = "source_current"
    _UID = "MST"
    _variable_label = "Source Current"
    _units = "µA"
    _priority = 0
    _value_range = (0, 20000)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        self._ipcon = IPConnection()  # Create IP connection
        self._iao = BrickletIndustrialAnalogOutV2(self._UID, self._ipcon)  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        super(IV_Current, self).__init__(*args, **kwargs)


    # Clean up measurement device.
    def __clean_up__(self):

        self._iao.set_enabled(False)

        self._ipcon.disconnect()

    # Disable voltage output
    def stop(self):
        self._iao.set_enabled(False)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        self._iao.set_current(self.get_value())
        self._iao.set_enabled(True)

    def clean_up(self):
        self.stop()