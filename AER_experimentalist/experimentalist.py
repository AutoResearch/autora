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
from AER_utils import Plot_Types
from abc import ABC, abstractmethod

class seed_strategy(Enum):
    UNIFORM = 1


class sample_strategy(Enum):
    ADVERSARIAL = 1
    UNCERTAINTY = 2

class Experimentalist(ABC):


    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None, seed_data_file="", experiment_design=None):

        self.experiment_id = 0
        self.conditions_per_experiment = exp_cfg.conditions_per_experiment
        self._experiment_sequence = None
        self._seed_parameters = [100]
        self._seed_strategy = seed_strategy.UNIFORM
        self._plots = dict()

        self.study_name = study_name
        self._experiment_server_host = experiment_server_host
        self._experiment_server_port = experiment_server_port
        self._experiment_design = experiment_design
        self.seed_data_file = seed_data_file


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
        self.set_experiment_id(0)

    def configure_experiment_client(self, experiment_server_host, experiment_server_port):
        self._experiment_server_host = experiment_server_host
        self._experiment_server_port = experiment_server_port

    def set_experiment_id(self, experiment_id):
        self.experiment_id = experiment_id

    def seed(self, object_of_study, datafile="", n=None):

        if n is None:
            n = exp_cfg.conditions_per_experiment

        # did the user specify a seed data file?
        if datafile == "" and self.seed_data_file != "":
            datafile = self.seed_data_file

        # seed with data file
        if datafile != "":
            file_path = AER_cfg.studies_folder + self.study_name + "/" + AER_cfg.experiment_folder + datafile

            if os.path.exists(file_path) is False:
                raise Exception("Seed data file does not exist: " + file_path)
            else:
                return self.load_experiment_data(object_of_study, file_path)

        else:
            if self._seed_strategy == seed_strategy.UNIFORM:
                # determine number of values for each independent variable for total number of data points assuming full counterbalancing
                n_variables = len(object_of_study.independent_variables)
                self._seed_parameters = [int(np.floor(n**(1/float(n_variables))))]
            else:
                raise Exception('Seed strategy not implemented.')

            experiment_file_path = self.generate_seed_experiment(object_of_study)
            return self.commission_experiment(object_of_study, experiment_file_path)

    def set_seed_strategy(self, strategy: seed_strategy, seed_parameters = list()):
        self._seed_strategy = strategy
        self._seed_parameters = seed_parameters

    def load_experiment_data(self, object_of_study, file_path):
        # read experiment file
        file = open(file_path, "r")
        for line in file:

            # read CSV
            raw_data = pandas.read_csv(file_path, header=0)
            col_names = raw_data.columns

            # generate dictionary that contains experiment sequence
            curated_data = dict()

            for IV in object_of_study.independent_variables:
                if IV.get_name() in col_names:
                    curated_data[IV.get_name()] = raw_data[IV.get_name()].tolist()
                else:
                    Exception('Could not find independent variable "' + IV.get_name() + '" in experiment data file: ' + file_path)

            for DV in object_of_study.dependent_variables:
                if DV.get_name() in col_names:
                    curated_data[DV.get_name()] = raw_data[DV.get_name()].tolist()
                else:
                    Exception('Could not find dependent variable "' + DV.get_name() + '" in experiment data file: ' + file_path)

            for CV in object_of_study.covariates:
                if CV.get_name() in col_names:
                    curated_data[CV.get_name()] = raw_data[CV.get_name()].tolist()
                else:
                    Exception('Could not find covariate "' + CV.get_name() + '" in experiment data file: ' + file_path)

            num_elements = len(curated_data[list(curated_data.keys())[0]])

            # add experiment label
            experiment_id_sequence = list()
            for i in range(num_elements):
                experiment_id_sequence.append(self.experiment_id)
            curated_data[object_of_study.key_experiment_id] = experiment_id_sequence

            return curated_data

    def generate_seed_experiment_default(self, object_of_study):

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

        return experiment_sequence

    def generate_seed_experiment(self, object_of_study):

        if self._experiment_design is None:
            experiment_sequence = self.generate_seed_experiment_default(object_of_study)
        else:
            experiment_sequence = self._experiment_design.generate(object_of_study)

        experiment_file_path = self._write_experiment(object_of_study, experiment_sequence)

        return experiment_file_path

    def _write_experiment(self, object_of_study, experiment_sequence):
        experiment_file_path = self._write_experiment_file(object_of_study)
        self._write_sequence_file(experiment_sequence)

        return experiment_file_path

    def _write_experiment_file(self, object_of_study, experiment_id = None):

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
        for idx, var in enumerate(object_of_study.independent_variables):
            f.write(var.get_name())
            if idx < len(object_of_study.independent_variables) - 1:
                f.write(',')
        f.write('\n')

        # write dependent variables
        f.write(exp_cfg.exp_file_DV_label)
        for idx, var in enumerate(object_of_study.dependent_variables):
            f.write(var.get_name())
            if idx < len(object_of_study.dependent_variables) - 1:
                f.write(',')
        f.write('\n')

        # write covariates
        f.write(exp_cfg.exp_file_CV_label)
        for idx, var in enumerate(object_of_study.covariates):
            f.write(var.get_name())
            if idx < len(object_of_study.covariates) - 1:
                f.write(',')
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

    def commission_experiment(self, object_of_study, experiment_file_path):

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
            data = self.load_experiment_data(object_of_study, data_file_path_dst)

            # add sequence and data to object of study
            return data

        else:
            raise Exception('Client could not retrieve experiment data from server.')

    def get_model_fit_plot_list(self, object_of_study):
        (IV_list_1, IV_list_2, DV_list) = object_of_study.get_plot_list()
        return (IV_list_1, IV_list_2, DV_list)

    def get_model_fit_plots(self, object_of_study, model):

        # get all possible plots
        (IV_list_1, IV_list_2, DV_list) = self.get_model_fit_plot_list(object_of_study)

        # for each plot
        for IV1, IV2, DV in zip(IV_list_1, IV_list_2, DV_list):

            IVs = [IV1, IV2]

            # generate model prediction
            resolution = 100
            counterbalanced_input = object_of_study.get_counterbalanced_input(resolution)
            if IV2 is None:  # prepare line plot
                x_prediction = object_of_study.get_IVs_from_input(counterbalanced_input, IV1)
            else:
                x_prediction = (object_of_study.get_IVs_from_input(counterbalanced_input, IV1), object_of_study.get_IVs_from_input(counterbalanced_input, IV2))
            y_prediction = model(counterbalanced_input)

            # get data points
            (input, output)  = object_of_study.get_dataset()
            if IV2 is None:  # prepare line plot
                x_data = object_of_study.get_IVs_from_input(input, IV1)
            else:
                x_data = (object_of_study.get_IVs_from_input(input, IV1), object_of_study.get_IVs_from_input(input, IV2))
            y_data = object_of_study.get_DV_from_output(output, DV)

            # get highlighted data points from last experiment
            last_experiment_id = object_of_study.get_last_experiment_id()
            (input_highlighted, output_highlighted) = object_of_study.get_dataset(experiment_id=last_experiment_id)
            if IV2 is None:  # prepare line plot
                x_data_highlighted = object_of_study.get_IVs_from_input(input_highlighted, IV1)
            else:
                x_data_highlighted = (object_of_study.get_IVs_from_input(input_highlighted, IV1),
                                      object_of_study.get_IVs_from_input(input_highlighted, IV2))
            y_data_highlighted = object_of_study.get_DV_from_output(output_highlighted, DV)

            # add x conditions

            if self._experiment_sequence is not None:
                x_conditions = list()
                experiment_sequence_rescaled = object_of_study.rescale_experiment_sequence(self._experiment_sequence)
                if IV2 is None:
                    for value in experiment_sequence_rescaled[IV1.get_name()]:
                        x_conditions.append(value)
                else:
                    for x1_value, x2_value in zip(experiment_sequence_rescaled[IV1.get_name()], experiment_sequence_rescaled[IV2.get_name()]):
                        x_conditions.append((x1_value, x2_value))
            else:
                x_conditions = None

            # determine y limits
            y_limit = [np.amin([np.amin(y_data.numpy()), np.amin(y_prediction.detach().numpy())]),
                       np.amax([np.amax(y_data.numpy()), np.amax(y_prediction.detach().numpy())])]

            # determine y_label
            y_label = DV.get_variable_label()

            # determine legend:
            legend = ['Data', 'Prediction', 'Novel Data']
            if self._experiment_sequence is not None:
                legend.append('Queried Data')

            # select data based on whether this is a line or a surface plot
            if IV2 is None: # prepare line plot

                # determine plot type
                type = Plot_Types.LINE_SCATTER
                plot_name = DV.get_name() + "(" + IV1.get_name() + ")"

                # determine x limits
                x_limit = object_of_study.get_variable_limits(IV1)

                # determine x_label
                x_label = IV1.get_variable_label()

            else: # prepare surface plot
                # determine plot type
                type = Plot_Types.SURFACE_SCATTER

                # determine x limits
                x_limit = (object_of_study.get_variable_limits(IV1),
                           object_of_study.get_variable_limits(IV2))

                # determine x_labels
                x_label = (IV1.get_variable_label(), IV2.get_variable_label())

            plot_dict = self._generate_plot_dict(type, x=x_data.detach().numpy(), y=y_data.detach().numpy(), x_limit=x_limit, y_limit=y_limit, x_label=x_label, y_label=y_label,
                                     legend=legend, image=None, x_model=x_prediction.detach().numpy(), y_model=y_prediction.detach().numpy(), x_highlighted=x_data_highlighted,
                                     y_highlighted=y_data_highlighted, x_conditions=x_conditions)
            self._plots[plot_name] = plot_dict


    def _generate_plot_dict(self, type, x, y, x_limit=None, y_limit=None, x_label=None, y_label=None, legend=None, image=None, x_model=None, y_model=None, x_highlighted=None, y_highlighted=None, x_conditions=None, y_conditions=None):
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
        if x_conditions is not None:
            plot_dict[AER_cfg.plot_key_x_conditions] = x_conditions

        return plot_dict

    def sample_experiment(self, model, object_of_study):

        self.init_experiment_search(model, object_of_study)

        for condition in self.conditions_per_experiment:
            for i in range(exp_cfg.max_num_rejections):
                trial = self.sample_experiment_condition(model, object_of_study, condition)
                if self._experiment_design is not None:
                    valid = self._experiment_design.validate_trial(object_of_study, trial, self._experiment_sequence)
                    if valid:
                        break
                else:
                    break
            for key in trial.keys():
                self._experiment_sequence[key].append(trial[key])

        experiment_file_path = self._write_experiment(object_of_study, self._experiment_sequence)

        return experiment_file_path

    # method for initializing experiment search
    @abstractmethod
    def init_experiment_search(self, model, object_of_study):
        # increment experiment id
        self.set_experiment_id(self.experiment_id+1)
        self._experiment_sequence = object_of_study.new_experiment_sequence()
        return

    # experiment search single condition
    @abstractmethod
    def sample_experiment_condition(self, model, object_of_study, condition):
        pass

    def get_plots(self, object_of_study, model):
        self.get_model_fit_plots(object_of_study, model)
        return self._plots
