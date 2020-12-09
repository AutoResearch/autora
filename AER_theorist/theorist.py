import os
import sys
import glob
import shutil
import logging
import AER_config as AER_cfg
from AER_theorist.theorist_GUI import Theorist_GUI
from tkinter import *

from abc import ABC, abstractmethod
from AER_theorist.object_of_study import Object_Of_Study

class Theorist(ABC):

    model_search_id = 0

    _cfg = None

    study_name = ""
    study_path = ""
    scripts_path = ""
    results_path = ""
    simulation_files = ""

    _meta_parameters = list()
    _meta_parameters_iteration = 0

    _plot = False

    _model_search_parameters = dict()
    _performance_plots = dict()
    _supplementary_plots = dict()

    _loss_plot_name = "Training & Validation Loss"
    _pattern_plot_name = "Predicted vs. Target Pattern"

    target_pattern = []
    prediction_pattern = []

    def __init__(self, study_name):

        self.model_search_id = 0
        self.study_name = study_name
        self.setup_simulation_directories()
        self.copy_scripts(scripts_to_save=glob.glob(self.simulation_files))

    def GUI(self, object_of_study):
        root = Tk()

        app = Theorist_GUI(theorist=self, object_of_study=object_of_study, root=root)

        root.mainloop()

    @abstractmethod
    def get_model_search_parameters(self):
        pass

    def set_model_search_parameter(self, key, str_value):
        if key in self._model_search_parameters.keys():
            if self._model_search_parameters[key][1] is True:
                self._model_search_parameters[key][0] = self._model_search_parameters[key][2](str_value)
                self.assign_model_search_parameters()
            else:
                raise Exception("Not allowed to modify model search parameter '" + str(key) + "'. Dictionary self._model_search_parameters indicates that the parameter cannot be modified.")
        else:
            raise Exception("Key '" + str(key) + "' not in dictionary self._model_search_parameters")

    @abstractmethod
    def assign_model_search_parameters(self):
        pass

    @abstractmethod
    def commission_meta_search(self, object_of_study):
        pass

    @abstractmethod
    def init_meta_search(self, object_of_study):
        self.model_search_id += 1
        self.setup_logging()

    @abstractmethod
    def search_model(self, object_of_study):
        pass

    @abstractmethod
    def init_model_search(self, object_of_study):
        pass

    @abstractmethod
    def run_model_search_epoch(self, object_of_study):
        pass

    @abstractmethod
    def log_model_search(self, object_of_study):
        pass

    @abstractmethod
    def plot_model(self, object_of_study):
        pass

    def get_performance_plots(self, object_of_study):
        self.update_loss_plot()
        self.update_pattern_plot()
        return self._performance_plots

    @abstractmethod
    def get_supplementary_plots(self, object_of_study):
        pass

    @abstractmethod
    def update_loss_plot(self):
        pass

    @abstractmethod
    def update_pattern_plot(self):
        pass

    def setup_simulation_directories(self):

        # study directory
        self.study_path = AER_cfg.studies_folder \
                          + self.study_name + "/"

        if not os.path.exists(self.study_path):
            os.mkdir(self.study_path)

        # scripts directory
        self.scripts_path = AER_cfg.studies_folder\
                            + self.study_name + "/"\
                            + AER_cfg.models_folder\
                            + AER_cfg.models_scripts_folder

        if not os.path.exists(self.scripts_path):
            os.mkdir(self.scripts_path)

        # results directory
        self.results_path = AER_cfg.studies_folder\
                            + self.study_name + "/"\
                            + AER_cfg.models_folder \
                            + AER_cfg.models_results_folder

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def copy_scripts(self, scripts_to_save=None):

        # copy scripts
        if scripts_to_save is not None:
            for script in scripts_to_save:
                dst_file = os.path.join(self.scripts_path, os.path.basename(script))
                shutil.copyfile(script, dst_file)


    def setup_logging(self):

        # determine the format for logging: event time and message
        log_format = '%(asctime)s %(message)s'
        # sets u a logging system in python,
        # - the stream is set to to the output console (stdout)
        # - report events that occur during normal operation of a program (logging.INFO), i.e. not during debugging
        # - use the pre-specified log format with time and message
        # - use the corresponding date format
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        # specify handle to file where logging output is stored
        fh = logging.FileHandler(os.path.join(self.results_path, 'log_' + str(self.model_search_id) + '.txt'))
        # specify format for logging
        fh.setFormatter(logging.Formatter(log_format))
        # adds file name to logger
        logging.getLogger().addHandler(fh)

    def _generate_plot_dict(self, type, x, y, x_limit=None, y_limit=None, x_label=None, y_label=None, legend=None, image=None, x_model=None, y_model=None, x_highlighted=None, y_highlighted=None):
        # generate plot dictionary
        plot_dict = dict()
        plot_dict[AER_cfg.plot_key_type] = type
        plot_dict[AER_cfg.plot_key_x_data] = x
        plot_dict[AER_cfg.plot_key_y_data] = y
        if x_limit is not None:
            plot_dict[AER_cfg.plot_key_x_limit] = x_limit
        if y_limit is not None:
            plot_dict[AER_cfg.plot_key_y_limit] = y_limit
        if x_label is not None:
            plot_dict[AER_cfg.plot_key_x_label] = x_label
        if y_label is not None:
            plot_dict[AER_cfg.plot_key_y_label] = y_label
        if legend is not None:
            plot_dict[AER_cfg.plot_key_legend] = legend
        if image is not None:
            plot_dict[AER_cfg.plot_key_image] = image
        if x_model is not None:
            plot_dict[AER_cfg.plot_key_x_model] = x_model
        if y_model is not None:
            plot_dict[AER_cfg.plot_key_y_model] = y_model
        if x_highlighted is not None:
            plot_dict[AER_cfg.plot_key_x_highlighted_data] = x_highlighted
        if y_highlighted is not None:
            plot_dict[AER_cfg.plot_key_y_highlighted_data] = y_highlighted

        return plot_dict