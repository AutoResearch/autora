from AER_experimentalist.experiment_environment.IV_time import IV_Time
from AER_experimentalist.experiment_environment.IV_trial import IV_Trial
from AER_experimentalist.experiment_environment.IV_voltage import IV_Voltage
from AER_experimentalist.experiment_environment.IV_current import IV_Current
from AER_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico
from AER_experimentalist.experiment_environment.DV_time import DV_Time
from AER_experimentalist.experiment_environment.DV_voltage import DV_Voltage
from AER_experimentalist.experiment_environment.DV_current import DV_Current
from AER_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico

IV_labels = {'time_IV': (IV_Time, 'Time', '', 'time_IV', 's', 1, (0, 3600)),
             'trial': (IV_Trial, 'Trial', '', 'trial', 'trials', 0, (0, 10000000)),
             'source_voltage': (IV_Voltage, 'Source Voltage', 'MST', 'source_voltage', 'mV', 2, (0, 5000)),
             'source_current': (IV_Current, 'Source Current', 'MST', 'source_current', 'µA', 2, (0, 20000)),
             'color_red': (IV_In_Silico, 'Color Unit Red', None, 'color_red', 'activation', 0, (0, 1)),
             'color_green': (IV_In_Silico, 'Color Unit Green', None, 'color_green', 'activation', 0, (0, 1)),
             'word_red': (IV_In_Silico, 'Word Unit Red', None, 'word_red', 'activation', 0, (0, 1)),
             'word_green': (IV_In_Silico, 'Word Unit Green', None, 'word_green', 'activation', 0, (0, 1)),
             'task_color': (IV_In_Silico, 'Task Unit Color Naming', None, 'task_color', 'activation', 0, (0, 1)),
             'task_word': (IV_In_Silico, 'Task Unit Word Reading', None, 'task_word', 'activation', 0, (0, 1)),
             'S1': (IV_In_Silico, 'Stimulus 1 Intensity', None, 'S1', 'activation', 0, (0, 5)),
             'S2': (IV_In_Silico, 'Stimulus 2 Intensity', None, 'S2', 'activation', 0, (0, 5)),
             'learning_trial': (IV_In_Silico, 'Trial', None, 'learning_trial', 'trial', 0, (0, 1000)),
             'P_initial': (IV_In_Silico, 'Initial Performance', None, 'P_initial', 'probability', 0, (0, 1)),
             'P_asymptotic': (IV_In_Silico, 'Best Performance', None, 'P_asymptotic', 'probability', 0, (0, 1)),
             'x1_lca': (IV_In_Silico, 'x1', None, 'x1_lca', 'net input', 0, (-1000, 1000)),
             'x2_lca': (IV_In_Silico, 'x2', None, 'x2_lca', 'net input', 0, (-1000, 1000)),
             'x3_lca': (IV_In_Silico, 'x3', None, 'x3_lca', 'net input', 0, (-1000, 1000)),
             }

DV_labels = {'time_DV': (DV_Time, 'Time', '', 'time_DV', 's', 0, (0, 3600)),
             'voltage0': (DV_Voltage, 'Voltage 0', 'MjY', 'voltage0', 'mV', 1, (-3500, 3500)),
             'voltage1': (DV_Voltage, 'Voltage 1', 'MjY', 'voltage1', 'mV', 1, (-3500, 3500)),
             'current0': (DV_Current, 'Current 0', 'Hfg', 'current0', 'mA', 2, (0, 20)),
             'current1': (DV_Current, 'Current 1', 'Hfg', 'current1', 'mA', 2, (0, 20)),
             'verbal_red': (DV_In_Silico, 'Verbal Response Red', None, 'verbal_red', 'activation', 0, (0, 1)),
             'verbal_green': (DV_In_Silico, 'Verbal Response Green', None, 'verbal_green', 'activation', 0, (0, 1)),
             'verbal_sample': (DV_In_Silico, 'Verbal Response Sample', None, 'verbal_sample', 'class', 0, (0, 1)),
             'difference_detected': (DV_In_Silico, 'Difference Detected', None, 'difference_detected', 'activation', 0, (0, 1)),
             'difference_detected_sample': (DV_In_Silico, 'Difference Detected', None, 'difference_detected_sample', 'class', 0, (0, 1)),
             'learning_performance': (DV_In_Silico, 'Accuracy', None, 'learning_performance', 'probability', 0, (0, 1)),
             'learning_performance_sample': (DV_In_Silico, 'Accuracy Sample', None, 'learning_performance_sample', 'class', 0, (0, 1)),
             'dx1_lca': (DV_In_Silico, 'dx1', None, 'dx1_lca', 'net input delta', 0, (-1000, 1000)),
             }

