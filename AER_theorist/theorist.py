import os
import sys
import glob
import shutil
import logging
import AER_config as AER_cfg

from abc import ABC, abstractmethod
from AER_theorist.object_of_study import Object_Of_Study

class Theorist(ABC):

    object_of_study = None
    model_search_id = 0

    _cfg = None

    study_name = ""
    study_path = ""
    scripts_path = ""
    results_path = ""
    simulation_files = ""

    def __init__(self, object_of_study: Object_Of_Study):

        self.object_of_study = Object_Of_Study
        self.study_name = self.object_of_study.get_name()

        self.model_search_id = 0
        self.setup_simulation_directories()
        self.copy_scripts(scripts_to_save=glob.glob(self.simulation_files))


    @abstractmethod
    def initialize_model_search(self):
        self.model_search_id += 1
        self.setup_logging()

    @abstractmethod
    def commission_model_search(self):
        pass

    @abstractmethod
    def run_model_search(self):
        pass


    def setup_simulation_directories(self):

        # study directory
        self.study_path = AER_cfg.studies_folder + self.study_name + "/"
        if not os.path.exists(self.study_path):
            os.mkdir(self.study_path)

        # scripts directory
        self.scripts_path = AER_cfg.studies_folder + self.study_name + "/" + AER_cfg.models_scripts_folder + "/"
        if not os.path.exists(self.scripts_path):
            os.mkdir(self.scripts_path)

        # results directory
        self.results_path = AER_cfg.studies_folder + self.study_name + "/" + AER_cfg.models_results_folder + "/"
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
        fh = logging.FileHandler(os.path.join(self.results_path, 'log_' + self.model_search_id + '.txt'))
        # specify format for logging
        fh.setFormatter(logging.Formatter(log_format))
        # adds file name to logger
        logging.getLogger().addHandler(fh)