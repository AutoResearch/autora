from variable_labels import IV_labels, DV_labels
from IV_trial import IV_Trial
from IV_time import IV_Time
from DV_time import DV_Time
import experiment_config as config
import time
import pandas
import AER_experimentalist.experimentalist_config as cfg

class Experiment():

    _path = ""
    _data_path = ""
    _sequences_folder = config.sequences_path
    _data_folder = config.data_path

    IVs = list()
    DVs = list()
    CVs = list()

    sequence = dict()   # stores experiment sequence (values for independent variables)
    data = dict()       # stores collected data (measurements for dependent variables)
    actual_trials = 0   # indicates current trial

    _IV_trial_idx = -1  # index of independent variable "trial"
    _IV_time_idx = -1   # index of independent variable "time"
    _DV_time_idx = -1   # index of dependent variable "time"

    _current_trial = 0

    _ITI = 2.0     # inter trial interval in seconds

    # Initialization requires path to experiment file
    def __init__(self, path, main_directory = ""):
        self._path = path
        self._main_dir = main_directory

        # initialize
        self.IVs = list()
        self.DVs = list()
        self.CVs = list()
        self.sequence = dict()
        self.data = dict()
        self._current_trial = 0
        self._IV_trial_idx = -1
        self._IV_time_idx = -1
        self._DV_time_idx = -1
        self._ITI = 1.0

        # read experiment file
        file = open(path, "r")
        for line in file:

            # read and print line of experiment file
            string = str(line)
            string = string.replace('\n', '')
            string = string.replace(' ', '')
            print(line)

            # read independent variables from line
            if(string.find(cfg.exp_file_IV_label) != -1):
                string = string.replace(cfg.exp_file_IV_label, '')
                labels = string.split(',')
                for idx, label in enumerate(labels):
                    if (label in IV_labels) is False:
                        raise Exception('Could not identify sequence variable: ' + label)
                    (IV_class, name, UID, variable_label, units, priority, value_range) = IV_labels.get(label)
                    # overwrite priority
                    priority = idx
                    self.IVs.append(IV_class(variable_label, UID, name, units, priority, value_range))

            # read dependent variables and covariates from line
            elif(string.find(cfg.exp_file_DV_label) != -1 or string.find(cfg.exp_file_CV_label) != -1):

                # determine whether this is a covariate (CV) or a dependent variable (DV)
                covariate = False
                if(string.find(cfg.exp_file_CV_label) != -1):
                    covariate = True
                    string = string.replace(cfg.exp_file_CV_label, '')
                else:
                    string = string.replace(cfg.exp_file_DV_label, '')

                # read in variables
                labels = string.split(',')
                for idx, label in enumerate(labels):
                    if (label in DV_labels) is False:
                        raise Exception('Could not identify sequence variable: ' + label)
                    (V_class, name, UID, variable_label, units, priority, value_range) = DV_labels.get(label)
                    # overwrite priority
                    priority = idx
                    if(covariate):
                        self.CVs.append(V_class(variable_label, UID, name, units, priority, value_range))
                        self.CVs(-1).set_covariate(True)
                    else:
                        self.DVs.append(V_class(variable_label, UID, name, units, priority, value_range))

            # read sequence file
            if (string.find(cfg.exp_file_sequence_label) != -1):
                string = string.replace(cfg.exp_file_sequence_label, '')
                csv_path = self._sequences_folder + string
                self.read_csv(self._main_dir + csv_path)

            # read data file path
            if (string.find(cfg.exp_file_data_label) != -1):
                string = string.replace(cfg.exp_file_data_label, '')
                self._data_path = self._data_folder + string

        # initialize experiment
        self.init_experiment()

    def read_csv(self, csv_path):

        # read CSV
        sequence_data = pandas.read_csv(csv_path, header=0)

        colnames = sequence_data.columns

        # generate dictionary that contains experiment sequence
        self.sequence = dict()

        for IV_name in colnames:
            seq = sequence_data[IV_name].tolist()
            self.sequence[IV_name] = seq

    # Initializes the experiment
    def init_experiment(self):

        # set up data
        for dependent_variable in self.DVs:
            self.data[dependent_variable.get_variable_label()] = list()

        # determine indices for independent variables trial and time
        for idx, independent_variable in enumerate(self.IVs):
            if isinstance(independent_variable, IV_Trial):
                self._IV_trial_idx = idx

            if isinstance(independent_variable, IV_Time):
                self._IV_time_idx = idx

        # initialize dependent variable time
        for idx, dependent_variable in enumerate(self.DVs):
            if isinstance(dependent_variable, DV_Time):
                self._DV_time_idx = idx

        # execute inter-trial interval
        self.ITI()

        # initialize trial and time variables
        if self._IV_trial_idx != -1:
            self.IVs[self._IV_trial_idx].set_value(0)

        if self._IV_time_idx != -1:
            self.IVs[self._IV_time_idx].reset()

        if self._DV_time_idx != -1:
            self.DVs[self._DV_time_idx].reset()

        # determine number of trials

        # Use trial variable as iteration variable if available,
        # otherwise generate iteration variable from first variable in dictionary.
        # Note that actual_trials refers to the number of actual measurements
        # (there could be multiple measurements per nominal trial).
        if self._IV_trial_idx != -1:
            trial_IV = self.IVs[self._IV_trial_idx]
            if self.sequence.get(trial_IV.get_variable_label()) is None:
                raise Exception("Could not find the 'trial' variable in the sequence.")
            self.actual_trials = range(0, len(self.sequence.get(trial_IV.get_variable_label())))

        else:
            # pick first variable in sequence and determine number of trials based on that
            self.actual_trials = range(0, len(self.sequence[0]))

    # Run the experiment
    def run_experiment(self):

        self.init_experiment()

        # run experiment
        for trial in self.actual_trials:

            self._current_trial = trial
            self.run_trial()

    # Runs a single experiment trial.
    def run_trial(self):

        trial = self._current_trial

        # set values for independent variables
        for independent_variable in self.IVs:
            # fetch value for variable
            independent_variable.set_value_from_dict(self.sequence, trial)

        # reset time stamp of trial_IV if new trial begins
        if self._IV_trial_idx != -1 and trial > 0:
            current_trial = self.sequence[self.IVs[self._IV_trial_idx].get_variable_label()][trial]
            last_trial = self.sequence[self.IVs[self._IV_trial_idx].get_variable_label()][trial-1]
            if current_trial != last_trial:
                self.ITI()
                if self._IV_time_idx != -1:
                    self.IVs[self._IV_time_idx].reset()

        # reset time stamp of trial_DV if new trial begins
        if self._DV_time_idx != -1 and self._IV_trial_idx != -1 and trial > 0:
            current_trial = self.sequence[self.IVs[self._IV_trial_idx].get_variable_label()][trial]
            last_trial = self.sequence[self.IVs[self._IV_trial_idx].get_variable_label()][trial-1]
            if current_trial != last_trial:
                self.DVs[self._DV_time_idx].reset()

        # after values are set, we can start to manipulate without significant delays
        for independent_variable in self.IVs:
            independent_variable.manipulate()

        # measure dependent variables
        for dependent_variable in self.DVs:
            dependent_variable.measure()
            self.data[dependent_variable.get_variable_label()].append(dependent_variable.get_value())

    # gets the current trial
    def set_current_trial(self, trial):
        self._current_trial = trial

    # returns the current trial number
    def get_current_trial(self):
        return self._current_trial

    # determine whether current actual trial marks the end of a nominal trial
    def is_end_of_trial(self):

        # return true if no trial variable exists
        if self._IV_trial_idx == -1:
            return True

        IV_trial = self.IVs[self._IV_trial_idx]
        trial_sequence = self.sequence.get(IV_trial.get_variable_label())

        # return true of the current actual trial marks the last in the sequence
        if self._current_trial >= (len(trial_sequence)-1):
            return True

        # return true if this trial marks the end of a nominal trial
        if trial_sequence[self._current_trial] != trial_sequence[self._current_trial+1]:
            return True
        else:
            return False

    def get_IV(self, variable_label):

        for IV in self.IVs:
            if IV.get_variable_label() ==  variable_label:
                return IV

        return None

    def get_DV_CV(self, variable_label):

        for DV in self.DVs:
            if DV.get_variable_label() == variable_label:
                return DV

        for CV in self.CVs:
            if CV.get_variable_label() == variable_label:
                return CV

        return None

    def get_IV_labels(self):
        names = list()
        for IV in self.IVs:
            names.append(IV.get_variable_label())

        return names

    def get_DV_labels(self):
        names = list()
        for DV in self.DVs:
            names.append(DV.get_variable_label())
        return names

    def get_CV_labels(self):
        names = list()
        for CV in self.CVs:
            names.append(CV.get_variable_label())
        return names

        # get list of current IV's
    def current_IVs_to_list(self, trial = None):

        if trial is None:
            trial = self._current_trial

        IV_description = list()

        for IV in self.IVs:
            IV_description.append((IV.get_name(), IV.get_value_from_dict(self.sequence, trial)))

        return IV_description

    # get list of current DV's
    def current_DVs_to_list(self, trial=None):

        if trial is None:
            trial = self._current_trial

        DV_description = list()

        for DV in self.DVs:
            DV_description.append((DV.get_variable_label(), DV.get_value()))

        for CV in self.CVs:
            DV_description.append((CV.get_variable_label(), CV.get_value()))

        return DV_description

    # get list of current CV's
    def current_CVs_to_list(self, trial=None):

        if trial is None:
            trial = self._current_trial

        CV_description = list()

        for CV in self.DVs:
            CV_description.append((CV.get_variable_label(), CV.get_value()))

        return CV_description

    def ITI(self):
        time.sleep(self._ITI)

    # writes current independent and dependent variables to csv file
    def data_to_csv(self, filepath=None):

        if filepath is not None:
            filepath = filepath
        else:
            filepath = self._data_path

        # generate data frame

        column_names = list()
        data = dict()

        # add dependent variables
        for IV in self.IVs:
            column_names.append(IV.get_variable_label())
            data[IV.get_variable_label()] = self.sequence[IV.get_variable_label()]
        pass

        # add independent variables
        for DV in self.DVs:
            column_names.append(DV.get_variable_label())
            data[DV.get_variable_label()] = self.data[DV.get_variable_label()]
        pass

        data_frame = pandas.DataFrame(data, columns = column_names)

        data_frame.to_csv(filepath)

    def clean_up(self):
        for IV in self.IVs:
            IV.clean_up()

        for DV in self.DVs:
            DV.clean_up()

        for CV in self.CVs:
            CV.clean_up()