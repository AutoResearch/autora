from IV_time import IV_Time
from IV_trial import IV_Trial
from IV_voltage import IV_Voltage
from IV_current import IV_Current
from DV_time import DV_Time
from DV_voltage import DV_Voltage
from DV_current import DV_Current

IV_labels = {'time_IV': (IV_Time, 'Time', '', 'time_IV', 's', 1, (0, 3600)),
             'trial': (IV_Trial, 'Trial', '', 'trial', 'trials', 0, (0, 10000000)),
             'source_voltage': (IV_Voltage, 'Source Voltage', 'MST', 'source_voltage', 'mV', 2, (0, 5000)),
             'source_current': (IV_Current, 'Source Current', 'MST', 'source_current', 'µA', 2, (0, 20000))
             }

DV_labels = {'time_DV': (DV_Time, 'Time', '', 'time_DV', 's', 0, (0, 3600)),
             'voltage0': (DV_Voltage, 'Voltage 0', 'MjY', 'voltage0', 'mV', 1, (-3500, 3500)),
             'voltage1': (DV_Voltage, 'Voltage 1', 'MjY', 'voltage1', 'mV', 1, (-3500, 3500)),
             'current0': (DV_Current, 'Current 0', 'Hfg', 'current0', 'mA', 2, (0, 20)),
             'current1': (DV_Current, 'Current 1', 'Hfg', 'current1', 'mA', 2, (0, 20)),
             }

