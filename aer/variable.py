import time
from abc import abstractmethod
from enum import Enum

import numpy as np
from tinkerforge.bricklet_industrial_analog_out_v2 import BrickletIndustrialAnalogOutV2
from tinkerforge.bricklet_industrial_dual_0_20ma_v2 import BrickletIndustrialDual020mAV2
from tinkerforge.bricklet_industrial_dual_analog_in_v2 import (
    BrickletIndustrialDualAnalogInV2,
)
from tinkerforge.ip_connection import IPConnection


class OutputTypes(Enum):
    REAL = 1
    SIGMOID = 2
    PROBABILITY = 3  # single probability
    PROBABILITY_SAMPLE = 4  # sample from single probability
    PROBABILITY_DISTRIBUTION = 5  # probability distribution over classes
    CLASS = 6  # sample from probability distribution over classes


class Variable:
    def __init__(
        self,
        name="",
        value_range=(0, 1),
        units="",
        type=OutputTypes.REAL,
        variable_label="",
        rescale=1,
        is_covariate=False,
        participant=None,
    ):

        self._name = name
        self._units = units
        self._value_range = value_range
        self._value = 0
        self.type = type
        if variable_label == "":
            self._variable_label = self._name
        else:
            self._variable_label = variable_label
        self._rescale = rescale
        self._is_covariate = is_covariate
        self._participant = participant

    def __get_value_range__(self):
        """Get range of variable.
        The variable range determines the minimum and maximum allowed value
        to be manipulated or measured."""
        return self._value_range

    def __set_value_range__(self, value_range):
        """Set range of variable.
        The variable range determines the minimum and maximum allowed value
        to be manipulated or measured."""
        self._value_range = value_range

    # Cap value of variable
    def __cap_value__(self, value):
        minimum = self._value_range[0]
        maximum = self._value_range[1]
        return np.min([np.max([value, minimum]), maximum])

    # Get value.
    def get_value(self):
        return self._value * self._rescale

    # Set value.
    def set_value(self, value):
        self._value = self.__cap_value__(value)

    def get_value_from_dict(self, dictionary, position):
        """Reads and sets value of independent variable from a dictionary with
        variable_label being the key"""

        value_list = dictionary.get(self.get_name())  # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception(
                "Could not find value with name '"
                + self.get_name()
                + "' in dictionary."
            )

        if position > len(value_list):
            raise Exception(
                "Queried position "
                + str(position)
                + " for variable "
                + self.get_name()
                + "'exceeds number of available positions for that variable in the dictionary."
            )

        return value_list[position] * self._rescale

    def get_value_list_from_dict(self, dictionary):
        value_list = dictionary.get(self.get_name())  # get_variable_label()

        if value_list is None:
            print(dictionary.keys())
            raise Exception(
                "Could not find value with name '"
                + self.get_name()
                + "' in dictionary."
            )

        rescaled_list = [element * self._rescale for element in value_list]

        return rescaled_list

    # Reads and sets value of independent variable from a dictionary
    # with variable_label being the key
    def set_value_from_dict(self, dictionary, position):

        value_list = dictionary.get(self.get_name())

        if value_list is None:
            raise Exception(
                "Could not find value with name '"
                + self.get_name()
                + "' in dictionary."
            )

        if position > len(value_list):
            raise Exception(
                "Queried position "
                + str(position)
                + " for variable "
                + self.get_name()
                + "'exceeds number of available positions for that variable in the dictionary."
            )

        self.set_value(value_list[position])

    # Get variable name.
    def get_name(self):
        return self._name

    # Set variable name.
    def set_name(self, name):
        self._name = name

    # Get variable units.
    def get_units(self):
        return self._units

    # Set variable units.
    def set_units(self, units):
        self._unitt = units

    # Get variable label.
    def get_variable_label(self):
        return self._variable_label

    # Set variable label.
    def set_variable_label(self, variable_label):
        self._variable_label = variable_label

    def set_covariate(self, is_covariate):
        """Set whether this dependent variable is treated as covariate."""
        self._is_covariate = is_covariate


class IV(Variable):
    pass


class DV(Variable):
    pass


class IVInSilico(IV):

    _variable_label = "IV"
    _name = "independent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        super(IVInSilico, self).__init__(*args, **kwargs)

    def manipulate(self):
        self._participant.set_value(self._name, self.get_value())


class DVInSilico(DV):

    _variable_label = "DV"
    _name = "dependent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):
        super(DVInSilico, self).__init__(*args, **kwargs)

    def measure(self):
        measurement = self._participant.get_value(self._name)
        self.set_value(measurement)


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

    def clean_up(self):
        pass


class IVTF(TinkerforgeVariable, IV):
    def __init__(self, *args, **kwargs):
        self._name = "IV"
        self._variable_label = "Independent Variable"

        super(IVTF, self).__init__(*args, **kwargs)

    # Method for measuring dependent variable.
    @abstractmethod
    def manipulate(self):
        pass

    # Method for cleaning up measurement device.
    @abstractmethod
    def __clean_up__(self):
        pass


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

    def __clean_up__(self):
        pass


class VTime:

    _t0 = 0

    def __init__(self):
        self._t0 = time.time()

    # Resets reference time.
    def reset(self):
        self._t0 = time.time()


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

    def __clean_up__(self):
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
    def __clean_up__(self):

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


class DVTF(TinkerforgeVariable, DV):
    def __init__(self, *args, **kwargs):
        self._name = "DV"
        self._variable_label = "Dependent Variable"

        self._is_covariate = False

        super(DVTF, self).__init__(*args, **kwargs)

    # Method for measuring dependent variable.
    @abstractmethod
    def measure(self):
        pass

    # Get whether this dependent variable is treated as covariate.
    def __is_covariate__(self):
        return self._is_covariate

    # Set whether this dependent variable is treated as covariate.
    def __set_covariate__(self, is_covariate):
        self._is_covariate = is_covariate


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


class DV_Current(DVTF):

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
        self._id020 = BrickletIndustrialDual020mAV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if self._name == "current1":
            self.channel = 1
        else:
            self.channel = 0

    # Clean up measurement device.
    def __clean_up__(self):

        self._ipcon.disconnect()

    # Waits until specified time has passed relative to reference time
    def measure(self):
        current = self._id020.get_current(self.channel)
        self.set_value(current / 1000000.0)


class DV_Voltage(DVTF):

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
        self._idai = BrickletIndustrialDualAnalogInV2(
            self._UID, self._ipcon
        )  # Create device object

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        if self._name == "voltage1":
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


DV_labels = {
    "time_DV": (DVTime, "Time", "", "time_DV", "s", 0, (0, 3600)),
    "voltage0": (DV_Voltage, "Voltage 0", "MjY", "voltage0", "mV", 1, (-3500, 3500)),
    "voltage1": (DV_Voltage, "Voltage 1", "MjY", "voltage1", "mV", 1, (-3500, 3500)),
    "current0": (DV_Current, "Current 0", "Hfg", "current0", "mA", 2, (0, 20)),
    "current1": (DV_Current, "Current 1", "Hfg", "current1", "mA", 2, (0, 20)),
    "verbal_red": (
        DVInSilico,
        "Verbal Response Red",
        None,
        "verbal_red",
        "activation",
        0,
        (0, 1),
    ),
    "verbal_green": (
        DVInSilico,
        "Verbal Response Green",
        None,
        "verbal_green",
        "activation",
        0,
        (0, 1),
    ),
    "verbal_sample": (
        DVInSilico,
        "Verbal Response Sample",
        None,
        "verbal_sample",
        "class",
        0,
        (0, 1),
    ),
    "difference_detected": (
        DVInSilico,
        "Difference Detected",
        None,
        "difference_detected",
        "activation",
        0,
        (0, 1),
    ),
    "difference_detected_sample": (
        DVInSilico,
        "Difference Detected",
        None,
        "difference_detected_sample",
        "class",
        0,
        (0, 1),
    ),
    "learning_performance": (
        DVInSilico,
        "Accuracy",
        None,
        "learning_performance",
        "probability",
        0,
        (0, 1),
    ),
    "learning_performance_sample": (
        DVInSilico,
        "Accuracy Sample",
        None,
        "learning_performance_sample",
        "class",
        0,
        (0, 1),
    ),
    "dx1_lca": (
        DVInSilico,
        "dx1",
        None,
        "dx1_lca",
        "net input delta",
        0,
        (-1000, 1000),
    ),
}


IV_labels = {
    "time_IV": (IVTime, "Time", "", "time_IV", "s", 1, (0, 3600)),
    "trial": (IVTrial, "Trial", "", "trial", "trials", 0, (0, 10000000)),
    "source_voltage": (
        IVVoltage,
        "Source Voltage",
        "MST",
        "source_voltage",
        "mV",
        2,
        (0, 5000),
    ),
    "source_current": (
        IVCurrent,
        "Source Current",
        "MST",
        "source_current",
        "µA",
        2,
        (0, 20000),
    ),
    "color_red": (
        IVInSilico,
        "Color Unit Red",
        None,
        "color_red",
        "activation",
        0,
        (0, 1),
    ),
    "color_green": (
        IVInSilico,
        "Color Unit Green",
        None,
        "color_green",
        "activation",
        0,
        (0, 1),
    ),
    "word_red": (
        IVInSilico,
        "Word Unit Red",
        None,
        "word_red",
        "activation",
        0,
        (0, 1),
    ),
    "word_green": (
        IVInSilico,
        "Word Unit Green",
        None,
        "word_green",
        "activation",
        0,
        (0, 1),
    ),
    "task_color": (
        IVInSilico,
        "Task Unit Color Naming",
        None,
        "task_color",
        "activation",
        0,
        (0, 1),
    ),
    "task_word": (
        IVInSilico,
        "Task Unit Word Reading",
        None,
        "task_word",
        "activation",
        0,
        (0, 1),
    ),
    "S1": (IVInSilico, "Stimulus 1 Intensity", None, "S1", "activation", 0, (0, 5)),
    "S2": (IVInSilico, "Stimulus 2 Intensity", None, "S2", "activation", 0, (0, 5)),
    "learning_trial": (
        IVInSilico,
        "Trial",
        None,
        "learning_trial",
        "trial",
        0,
        (0, 1000),
    ),
    "P_initial": (
        IVInSilico,
        "Initial Performance",
        None,
        "P_initial",
        "probability",
        0,
        (0, 1),
    ),
    "P_asymptotic": (
        IVInSilico,
        "Best Performance",
        None,
        "P_asymptotic",
        "probability",
        0,
        (0, 1),
    ),
    "x1_lca": (IVInSilico, "x1", None, "x1_lca", "net input", 0, (-1000, 1000)),
    "x2_lca": (IVInSilico, "x2", None, "x2_lca", "net input", 0, (-1000, 1000)),
    "x3_lca": (IVInSilico, "x3", None, "x3_lca", "net input", 0, (-1000, 1000)),
}
