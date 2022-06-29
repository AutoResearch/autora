import time
from abc import abstractmethod

from tinkerforge.bricklet_industrial_analog_out_v2 import BrickletIndustrialAnalogOutV2
from tinkerforge.bricklet_industrial_dual_0_20ma_v2 import BrickletIndustrialDual020mAV2
from tinkerforge.bricklet_industrial_dual_analog_in_v2 import (
    BrickletIndustrialDualAnalogInV2,
)
from tinkerforge.ip_connection import IPConnection

from aer.variable import DV, IV, Variable
from aer.variable.time import VTime


class TinkerforgeVariable(Variable):

    _variable_label = ""
    _UID = ""
    _priority = 0

    def __init__(
        self,
        variable_label="",
        UID="",
        name="",
        units="",
        priority="",
        value_range=(0, 1),
        type=float,
    ):

        super().__init__(
            name=name,
            value_range=value_range,
            units=units,
            type=type,
            variable_label=variable_label,
        )

        self._UID = UID
        self._priority = priority

    # Get priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __get_priority__(self):
        return self._priority

    # Set priority of variable.
    # The priority is used to determine the sequence of variables to be measured or manipulated.
    def __set_priority__(self, priority):
        self._priority = priority

    @abstractmethod
    def clean_up(self):
        """Clean up measurement device."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from up measurement device."""
        pass


class IVTF(IV, TinkerforgeVariable):
    def __init__(self, *args, **kwargs):
        IV.__init__(self, *args, **kwargs)
        TinkerforgeVariable.__init__(self, *args, **kwargs)


class DVTF(DV, TinkerforgeVariable):
    def __init__(self, *args, **kwargs):
        DV.__init__(self, *args, **kwargs)
        TinkerforgeVariable.__init__(self, *args, **kwargs)


class IVTrial(IVTF):

    _name = "trial"
    _UID = ""
    _variable_label = "Trial"
    _units = "trials"
    _priority = 0
    _value_range = (0, 10000000)
    _value = 0

    def __init__(self, *args, **kwargs):
        super(IVTrial, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        pass

    def disconnect(self):
        pass


class IVTime(IVTF, VTime):

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
        super(IVTime, self).__init__(*args, **kwargs)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):

        t_wait = self.get_value() - (time.time() - self._t0)
        if t_wait <= 0:
            return
        else:
            time.sleep(t_wait)

    def disconnect(self):
        pass


class IVCurrent(IVTF):

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
        self._iao = BrickletIndustrialAnalogOutV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        super(IVCurrent, self).__init__(*args, **kwargs)

    # Clean up measurement device.
    def disconnect(self):

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


class IVVoltage(IVTF):

    _variable_label = "Source Voltage"
    _UID = "MST"
    _name = "source_voltage"
    _units = "mV"
    _priority = 0
    _value_range = (0, 5000)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        self._ipcon = IPConnection()  # Create IP connection
        self._iao = BrickletIndustrialAnalogOutV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        super(IVVoltage, self).__init__(*args, **kwargs)

    # Clean up measurement device.
    def disconnect(self):

        self._iao.set_enabled(False)

        self._ipcon.disconnect()

    # Disable voltage output
    def stop(self):
        self._iao.set_enabled(False)

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        self._iao.set_voltage(self.get_value())
        self._iao.set_enabled(True)

    def clean_up(self):
        self.stop()


class DVTime(DVTF, VTime):

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
        print(self._variable_label)
        super(DVTime, self).__init__(*args, **kwargs)
        print(self._variable_label)

    # Measure number of seconds relative to reference time
    def measure(self):

        value = time.time() - self._t0
        self.set_value(value)


class DVCurrent(DVTF):

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

        super(DVCurrent, self).__init__(*args, **kwargs)

        self._ipcon = IPConnection()  # Create IP connection
        self._id020 = BrickletIndustrialDual020mAV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if self._name == "current1":
            self.channel = 1
        else:
            self.channel = 0

    # Clean up measurement device.
    def disconnect(self):

        self._ipcon.disconnect()

    # Waits until specified time has passed relative to reference time
    def measure(self):
        current = self._id020.get_current(self.channel)
        self.set_value(current / 1000000.0)


class DVVoltage(DVTF):

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

        super(DVVoltage, self).__init__(*args, **kwargs)

        self._ipcon = IPConnection()  # Create IP connection
        self._idai = BrickletIndustrialDualAnalogInV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if self._name == "voltage1":
            self.channel = 1
        else:
            self.channel = 0

    # Clean up measurement device.
    def disconnect(self):
        self._ipcon.disconnect()

    # Waits until specified time has passed relative to reference time
    def measure(self):
        value = self._idai.get_voltage(self.channel)
        self.set_value(value)
