from abc import abstractmethod
from typing import Any, Tuple

from tinkerforge.bricklet_industrial_analog_out_v2 import BrickletIndustrialAnalogOutV2
from tinkerforge.bricklet_industrial_dual_0_20ma_v2 import BrickletIndustrialDual020mAV2
from tinkerforge.bricklet_industrial_dual_analog_in_v2 import (
    BrickletIndustrialDualAnalogInV2,
)
from tinkerforge.ip_connection import IPConnection
from variable import ValueType

from autora.variable import DV, IV, Variable


class TinkerforgeVariable(Variable):
    """
    A representation of a variable used in the Tinkerforge environment.
    """

    _variable_label = ""
    _UID = ""
    _priority = 0

    def __init__(
        self,
        variable_label: str = "",
        UID: str = "",
        name: str = "",
        units: str = "",
        priority: int = 0,
        value_range: Tuple[Any, Any] = (0, 1),
        type: ValueType = float,
    ):
        """
        Initializes a Tinkerforge variable.
        Args:
            variable_label: the label of the variable
            UID: the user identification of the variable
            name: the name of the variable
            units: the units of the variable
            priority: the priority of the variable
            value_range: the value range of the variable
            type: the type of the variable
        """

        super().__init__(
            name=name,
            value_range=value_range,
            units=units,
            type=type,
            variable_label=variable_label,
        )

        self._UID = UID
        self._priority = priority

    def __get_priority__(self) -> int:
        """
        Get priority of variable. The priority is used to determine the sequence of variables
        to be measured or manipulated.

        Returns:
            The priority of the variable.
        """
        return self._priority

    def __set_priority__(self, priority: int = 0):
        """
        Set priority of variable.
        The priority is used to determine the sequence of variables to be measured or manipulated.

        Arguments:
            priority: The priority of the variable.
        """
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
    """
    A representation of an independent variable used in the Tinkerforge environment.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an independent variable used in the Tinkerforge environment.

        For arguments, see [autora.variable.tinkerforge.TinkerforgeVariable]
        [autora.variable.tinkerforge.TinkerforgeVariable.__init__]
        """
        IV.__init__(self, *args, **kwargs)
        TinkerforgeVariable.__init__(self, *args, **kwargs)


class DVTF(DV, TinkerforgeVariable):
    """
    A representation of a dependent variable used in the Tinkerforge environment.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a dependent variable used in the Tinkerforge environment.

        For arguments, see [autora.variable.tinkerforge.TinkerforgeVariable]
        [autora.variable.tinkerforge.TinkerforgeVariable.__init__]
        """
        DV.__init__(self, *args, **kwargs)
        TinkerforgeVariable.__init__(self, *args, **kwargs)


class IVCurrent(IVTF):
    """
    An independent tinkerforge variable representing the current.
    """

    _name = "source_current"
    _UID = "MST"
    _variable_label = "Source Current"
    _units = "µA"
    _priority = 0
    _value_range = (0, 20000)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223

    def __init__(self, *args, **kwargs):
        """
        Initializes Industrial Analog Out 2.0 device.

        For arguments, see [autora.variable.tinkerforge.TinkerforgeVariable]
        [autora.variable.tinkerforge.TinkerforgeVariable.__init__]
        """

        self._ipcon = IPConnection()  # Create IP connection
        self._iao = BrickletIndustrialAnalogOutV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        super(IVCurrent, self).__init__(*args, **kwargs)

    def disconnect(self):
        """
        Disconnect from up measurement device.
        """

        self._iao.set_enabled(False)

        self._ipcon.disconnect()

    def stop(self):
        """
        Disable current output
        """

        self._iao.set_enabled(False)

    def manipulate(self):
        """
        Sets the current output to the specified value.
        """
        self._iao.set_current(self.get_value())
        self._iao.set_enabled(True)

    def clean_up(self):
        """
        Clean up measurement device.
        """
        self.stop()


class IVVoltage(IVTF):
    """
    An independent tinkerforge variable representing the voltage.
    """

    _variable_label = "Source Voltage"
    _UID = "MST"
    _name = "source_voltage"
    _units = "mV"
    _priority = 0
    _value_range = (0, 5000)
    _value = 0

    _HOST = "localhost"
    _PORT = 4223

    def __init__(self, *args, **kwargs):
        """
        Initializes Industrial Analog Out 2.0 device.
        """

        self._ipcon = IPConnection()  # Create IP connection
        self._iao = BrickletIndustrialAnalogOutV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        super(IVVoltage, self).__init__(*args, **kwargs)

    def disconnect(self):
        """
        Disconnect from up measurement device.
        """

        self._iao.set_enabled(False)

        self._ipcon.disconnect()

    def stop(self):
        """
        Disable voltage output
        """
        self._iao.set_enabled(False)

    def manipulate(self):
        """
        Sets the voltage output to the specified value.
        """
        self._iao.set_voltage(self.get_value())
        self._iao.set_enabled(True)

    def clean_up(self):
        """
        Clean up measurement device.
        """
        self.stop()


class DVCurrent(DVTF):
    """
    A dependent tinkerforge variable representing the current.
    """

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

    def __init__(self, *args, **kwargs):
        """
        Initializes Industrial Analog Out 2.0 device.

        For arguments, see [autora.variable.tinkerforge.TinkerforgeVariable]
        [autora.variable.tinkerforge.TinkerforgeVariable.__init__]
        """

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

    def disconnect(self):
        """
        Disconnect from up measurement device.
        """

        self._ipcon.disconnect()

    def measure(self):
        """
        Measures the current.
        """
        current = self._id020.get_current(self.channel)
        self.set_value(current / 1000000.0)


class DVVoltage(DVTF):
    """
    A dependent tinkerforge variable representing the voltage.
    """

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

    def __init__(self, *args, **kwargs):
        """
        Initializes Industrial Analog Out 2.0 device.

        For arguments, see [autora.variable.tinkerforge.TinkerforgeVariable]
        [autora.variable.tinkerforge.TinkerforgeVariable.__init__]
        """

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

    def disconnect(self):
        """
        Disconnect from up measurement device.
        """
        self._ipcon.disconnect()

    def measure(self):
        """
        Measures the voltage.
        """
        value = self._idai.get_voltage(self.channel)
        self.set_value(value)
