import AER_config as AER_cfg
import os
import shutil
import csv
import pandas
import numpy as np
import AER_experimentalist.experimentalist_config as exp_cfg
from AER_experimentalist.experiment_environment.experiment_client import Experiment_Client
from sweetpea.primitives import Factor
from sweetpea import fully_cross_block, synthesize_trials_non_uniform
from enum import Enum

class seed_strategy(Enum):
    UNIFORM = 1
    PROBABILITY = 2
    PROBABILITY_DISTRIBUTION = 3
    CLASS = 4


class sample_strategy(Enum):
    ADVERSARIAL = 1
    UNCERTAINTY = 2

class Experimantalist():

    study_name = "Default"
    experiment_id = 0

    _seed_strategy = seed_strategy.UNIFORM
    _seed_parameters = [100]

    _experiments_path = ""

    _experiment_server_host = None
    _experiment_server_port = None

    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None):

        self.study_name = study_name
        self._experiment_server_host = experiment_server_host
        self._experiment_server_port = experiment_server_port

        # generate folder for this study (if none exists yet)
        study_path = AER_cfg.studies_folder + study_name + "/"
        if not os.path.exists(study_path):
            os.mkdir(study_path)
        if not os.path.exists(study_path + AER_cfg.experiment_folder):
            os.mkdir(study_path + AER_cfg.experiment_folder)
        if not os.path.exists(study_path + AER_cfg.models_folder):
            os.mkdir(study_path + AER_cfg.models_folder)
        print('Experiment dir : {}'.format(study_path))

        self._experiments_path = study_path + AER_cfg.experiment_folder

    def configure_experiment_client(self, experiment_server_host, experiment_server_port):
        self._experiment_server_host = experiment_server_host
        self._experiment_server_port = experiment_server_port

    def set_experiment_id(self, experiment_id):
        self.experiment_id = experiment_id

    def seed(self, object_of_study, datafile=""):

        # seed with data file
        if datafile != "":
            file_path = AER_cfg.studies_folder + self.study_name + ''

            if os.path.exists(file_path) is False:
                Exception("Seed data file does not exist: " + file_path)
            else:
                return self.load_experiment_data(object_of_study, file_path, experiment_id=0)

        else:
            self.set_experiment_id(0)
            experiment_file_path = self.generate_seed_experiment(object_of_study)
            return self.commission_experiment(experiment_file_path)

    def set_seed_strategy(self, strategy: seed_strategy, seed_parameters = list()):
        self._seed_strategy = strategy
        self._seed_parameters = seed_parameters

    def load_experiment_data(self, object_of_study, file_path):
        # read experiment file
        file = open(file_path, "r")
        for line in file:

            # read CSV
            data = pandas.read_csv(file_path, header=0)
            col_names = data.columns

            # generate dictionary that contains experiment sequence
            data = dict()

            for IV in object_of_study.independent_variables:
                if IV.get_name() in data:
                    data[IV.get_name()] = data[IV.get_name()].tolist()
                else:
                    Exception('Could not find independent variable "' + IV.get_name() + '" in experiment data file: ' + file_path)

            for DV in object_of_study.dependent_variables:
                if DV.get_name() in data:
                    data[DV.get_name()] = data[DV.get_name()].tolist()
                else:
                    Exception('Could not find dependent variable "' + DV.get_name() + '" in experiment data file: ' + file_path)

            for CV in object_of_study.covariates:
                if CV.get_name() in data:
                    data[CV.get_name()] = data[CV.get_name()].tolist()
                else:
                    Exception('Could not find covariate "' + CV.get_name() + '" in experiment data file: ' + file_path)

            num_elements = len(data[0])

            # add experiment label
            experiment_id_sequence = list()
            for i in range(num_elements):
                experiment_id_sequence.append(self.experiment_id)
            data[AER_cfg.experiment_label] = experiment_id_sequence

            return data

    def generate_seed_experiment(self, object_of_study):

        experiment_design = list()

        # Uniform sampling
        if self._seed_strategy == seed_strategy.UNIFORM:

            # determine tested values for each independent variable
            resolution = self._seed_parameters[0]

            for var in object_of_study.independent_variables:
                factor = Factor(var.get_name(),
                                np.linspace(var._value_range[0], var._value_range[1], resolution).tolist())
                experiment_design.append(factor)
        else:
            Exception("Chosen seed strategy not implemented. Try: set_seed_strategy(seed_strategy.UNIFORM)")

        # generated crossed experiment with SweetPea
        block = fully_cross_block(experiment_design, experiment_design, [])
        experiment_sequence = synthesize_trials_non_uniform(block, 1)[0]

        experiment_file_path = self._write_experiment(object_of_study, experiment_sequence)

        return experiment_file_path

    def _write_experiment(self, object_of_study, experiment_sequence):
        experiment_file_path = self._write_experiment_file()
        self._write_sequence_file(object_of_study, experiment_sequence)

        return experiment_file_path

    def _write_experiment_file(self, object_of_study):

        # specify names
        experiment_file_path = self._experiments_path + AER_cfg.experiment_file_prefix + str(self.experiment_id) + '.exp'
        sequence_file_name = AER_cfg.experiment_file_prefix + str(
            self.experiment_id) + '_' + AER_cfg.sequence_file_suffix + '.csv'
        data_file_name = AER_cfg.experiment_file_prefix + str(
            self.experiment_id) + '_' + AER_cfg.data_file_suffix + '.csv'

        # open file
        f = open(experiment_file_path, 'w')

        # write independent variables
        f.write(exp_cfg.exp_file_IV_label)
        for var in object_of_study.independent_variables:
            f.write(var.get_name())
        f.write('\n')

        # write dependent variables
        f.write(exp_cfg.exp_file_DV_label)
        for var in object_of_study.dependent_variables:
            f.write(var.get_name())
        f.write('\n')

        # write covariates
        f.write(exp_cfg.exp_file_CV_label)
        for var in object_of_study.covariates:
            f.write(var.get_name())
        f.write('\n')

        # write sequence file name
        f.write(exp_cfg.exp_file_sequence_label + sequence_file_name + '\n')

        # write data file name
        f.write(exp_cfg.exp_file_data_label + data_file_name + '\n')

        # close file
        f.close()

        return experiment_file_path

    def _write_sequence_file(self, experiment_sequence):
        sequence_file_name = AER_cfg.experiment_file_prefix + str(
            self.experiment_id) + '_' + AER_cfg.sequence_file_suffix + '.csv'
        sequence_file_path = self._experiments_path + sequence_file_name

        # open file
        csv_columns = list(experiment_sequence.keys())
        num_rows = len(experiment_sequence[csv_columns[0]])

        try:
            with open(sequence_file_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(csv_columns)
                for row_idx in range(num_rows):
                    row = list()
                    for column in csv_columns:
                        row.append(experiment_sequence[column][row_idx])
                    writer.writerow(row)
        except IOError:
            print("I/O error occured while writing sequence file: " + sequence_file_path)

    def commission_experiment(self, experiment_file_path):

        # check if client is configured
        if self._experiment_server_host is None:
            raise Exception("Experiment server host not configured. Please specify server host using: configure_experiment_client(experiment_server_host, experiment_server_port)")

        if self._experiment_server_port is None:
            raise Exception("Experiment server port not configured. Please specify server port using: configure_experiment_client(experiment_server_host, experiment_server_port)")

        # get exeriment file name
        experiment_file_name = os.path.basename(experiment_file_path)

        # extract sequence file path from experiment file
        file = open(experiment_file_path, "r")
        for line in file:
            # read and print line of experiment file
            string = str(line)
            string = string.replace('\n', '')
            string = string.replace(' ', '')

            # read sequence file name
            if (string.find(exp_cfg.exp_file_sequence_label) != -1):
                sequence_file_name = string.replace(exp_cfg.exp_file_sequence_label, '')
                sequence_file_path = self._experiments_path + sequence_file_name

            # read data file name
            if (string.find(exp_cfg.exp_file_data_label) != -1):
                data_file_name = string.replace(exp_cfg.exp_file_data_label, '')

        # copy experiment file and sequence file to experiment client
        if os.path.exists(experiment_file_path):
            experiment_file_destination_path = exp_cfg.client_experiments_path + experiment_file_name
            shutil.copy(experiment_file_path, experiment_file_destination_path)
        else:
            raise Exception("Could not find experiment file: " + experiment_file_path)

        if os.path.exists(sequence_file_path):
            sequence_file_destination_path = exp_cfg.client_sequences_path + sequence_file_name
            shutil.copy(sequence_file_path, sequence_file_destination_path)
        else:
            raise Exception("Could not find sequence file: " + experiment_file_path)

        # launch experiment client
        session_ID = self.experiment_id
        if session_ID == 0:
            clear_sessions = True
        else:
            clear_sessions = False
        host = self._experiment_server_host
        port = self._experiment_server_port
        exp_client = Experiment_Client(session_ID, host=host, port=port)
        client_success = exp_client.submit_job(experiment_file_name, clear_sessions=clear_sessions)

        if client_success:
            # copy data from client
            data_file_path_src = exp_cfg.client_data_path + data_file_name

            if os.path.exists(data_file_path_src):
                data_file_name = AER_cfg.experiment_file_prefix + str(
                    self.experiment_id) + '_' + AER_cfg.data_file_suffix + '.csv'
                data_file_path_dst = self._experiments_path + data_file_name
                shutil.copy(data_file_path_src, data_file_path_dst)
            else:
                raise Exception("Could not find data file: " + data_file_path_src)

            # read data
            data = self.load_experiment_data(data_file_path_dst)

            # add sequence and data to object of study
            return data

        else:
            raise Exception('Client could not retrieve experiment data from server.')


    def sample_experiment(self, model, object_of_study):

        # increment experiment id
        self.set_experiment_id(self.experiment_id+1)

        # todo: implement

        # return experiment_file_path
        pass

