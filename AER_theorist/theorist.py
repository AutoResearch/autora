import os
import sys
import glob
import shutil
import logging
import AER_config as aer_config

from AER_utils import Plot_Types
from AER_theorist.theorist_GUI import Theorist_GUI
from tkinter import *
from matplotlib.figure import Figure


from abc import ABC, abstractmethod
from AER_theorist.object_of_study import Object_Of_Study

class Theorist(ABC):

    _loss_plot_name = "Training & Validation Loss"
    _pattern_plot_name = "Target vs. Predicted Pattern"

    def __init__(self, study_name):

        self._cfg = None

        self.study_name = ""
        self.study_path = ""
        self.scripts_path = ""
        self.results_path = ""
        self.results_plots_path = ""
        self.simulation_files = ""

        self._meta_parameters = list()
        self._meta_parameters_iteration = 0
        self._eval_meta_parameters_iteration = 0

        self.plot = False

        self._model_search_parameters = dict()
        self._performance_plots = dict()
        self._supplementary_plots = dict()
        self._validation_sets = dict()

        self.target_pattern = []
        self.prediction_pattern = []

        self.model_search_epochs = 100
        self.eval_epochs = 100

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

    def search_model(self, object_of_study):
        # initialize model search
        self.init_meta_search(object_of_study)

        # perform architecture search for different hyper-parameters
        for meta_params in self._meta_parameters:
            self.init_model_search(object_of_study)
            for epoch in range(self.model_search_epochs):
                self.run_model_search_epoch(epoch)
                if self.plot:
                    self.log_plot_data(epoch, object_of_study)
            self.log_model_search(object_of_study)
            if self.plot:
                self.plot_model_search(object_of_study)
            self.evaluate_model_search(object_of_study)
            self._meta_parameters_iteration += 1

        return self.get_best_model(object_of_study, plot_model=True)

    def evaluate_model_search(self, object_of_study):

        # initialize model search
        self.init_meta_evaluation(object_of_study)

        # perform architecture search for different hyper-parameters
        for eval_meta_params in self._eval_meta_parameters:
            self.init_model_evaluation(object_of_study)
            # loop over epochs
            for epoch in range(self.eval_epochs):
                logging.info('epoch %d', epoch)
                # run single epoch
                self.run_eval_epoch(epoch, object_of_study)
                # log performance (for plotting purposes)
                self.log_plot_data(epoch, object_of_study)

            # plot evaluation
            if self.plot:
                self.plot_model_eval(object_of_study)

            # log model evaluation
            self.log_model_evaluation(object_of_study)
            self._eval_meta_parameters_iteration += 1


        # sum up meta evaluation
        self.log_meta_evaluation(object_of_study)

    def plot_model_search(self, object_of_study):
        plot_label = self._meta_parameters_to_str()
        self.plot(object_of_study, plot_label)

    def plot_model_eval(self, object_of_study):
        plot_label = self._meta_parameters_to_str() + "_" + self._eval_meta_parameters_to_str()
        self.plot(object_of_study, plot_label)

    def plot(self, object_of_study, plot_label):
        performance_plots = self.get_performance_plots(object_of_study)
        self.save_plots(performance_plots, plot_label)
        supplementary_plots = self.get_supplementary_plots(object_of_study)
        self.save_plots(supplementary_plots, plot_label)

    def save_plots(self, plot_list, plot_label):

        for key in plot_list.keys():
            plot_dict = plot_list[key]

            type = plot_dict[aer_config.plot_key_type]

            plot_fig = Figure(figsize=(7, 7), dpi=100)

            if type == Plot_Types.SURFACE_SCATTER:
                plot_axis = plot_fig.add_subplot(111, projection='3d')
            else:
                plot_axis = plot_fig.add_subplot(111)

            # plot_fig.subplots_adjust(bottom=0.2)
            # plot_fig.subplots_adjust(left=0.35)

            if type == Plot_Types.LINE:

                # get relevant data
                x_data = plot_dict[aer_config.plot_key_x_data]
                y_data = plot_dict[aer_config.plot_key_y_data]
                x_limit = plot_dict[aer_config.plot_key_x_limit]
                y_limit = plot_dict[aer_config.plot_key_y_limit]
                x_label = plot_dict[aer_config.plot_key_x_label]
                y_label = plot_dict[aer_config.plot_key_y_label]
                legend = plot_dict[aer_config.plot_key_legend]

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]  # remove previous lines
                plots = list()
                for idx, (x, y, leg) in enumerate(zip(x_data, y_data, legend)):
                    plots.append(plot_axis.plot(x, y, aer_config.plot_colors[idx], label=leg))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=aer_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=aer_config.font_size)

                plot_axis.legend(loc=2, fontsize="small")


            elif type == Plot_Types.IMAGE:

                # get relevant data
                image = plot_dict[aer_config.plot_key_image]
                x_data = plot_dict[aer_config.plot_key_x_data]
                y_data = plot_dict[aer_config.plot_key_y_data]
                x_label = plot_dict[aer_config.plot_key_x_label]
                y_label = plot_dict[aer_config.plot_key_y_label]

                # generate image
                plot_axis.cla()
                plot_axis.imshow(image, interpolation='nearest', aspect='auto')
                x = x_data
                y = y_data
                plot_axis.plot(x, y, color='red')

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=aer_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=aer_config.font_size)

            elif type == Plot_Types.LINE_SCATTER:

                # get relevant data
                x_data = plot_dict[aer_config.plot_key_x_data]
                y_data = plot_dict[aer_config.plot_key_y_data]
                x_model = plot_dict[aer_config.plot_key_x_model]
                y_model = plot_dict[aer_config.plot_key_y_model]
                x_limit = plot_dict[aer_config.plot_key_x_limit]
                y_limit = plot_dict[aer_config.plot_key_y_limit]
                x_label = plot_dict[aer_config.plot_key_x_label]
                y_label = plot_dict[aer_config.plot_key_y_label]
                legend = plot_dict[aer_config.plot_key_legend]

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]  # remove previous lines
                plots = list()
                # plot data
                plots.append(plot_axis.scatter(x_data, y_data, marker='.', c='r', label=legend[0]))

                # plot model prediction
                plots.append(plot_axis.plot(x_model, y_model, 'k', label=legend[1]))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=aer_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=aer_config.font_size)

                plot_axis.legend(loc=2, fontsize="small")

            elif type == Plot_Types.SURFACE_SCATTER:

                # get relevant data
                (x1_data, x2_data) = plot_dict[aer_config.plot_key_x_data]
                y_data = plot_dict[aer_config.plot_key_y_data]
                (x1_model, x2_model) = plot_dict[aer_config.plot_key_x_model]
                y_model = plot_dict[aer_config.plot_key_y_model]
                (x1_limit, x2_limit) = plot_dict[aer_config.plot_key_x_limit]
                y_limit = plot_dict[aer_config.plot_key_y_limit]
                (x1_label, x2_label) = plot_dict[aer_config.plot_key_x_label]
                y_label = plot_dict[aer_config.plot_key_y_label]
                legend = plot_dict[aer_config.plot_key_legend]

                # generate plots
                plot_axis.cla()
                plots = list()

                # plot data
                plots.append(plot_axis.scatter(x1_data, x2_data, y_data, color=(1, 0, 0, 0.5), label=legend[0]))
                # plot model prediction
                plots.append(
                    plot_axis.plot_trisurf(x1_model, x2_model, y_model, color=(0, 0, 0, 0.5), label=legend[1]))

                # adjust axes
                plot_axis.set_xlim(x1_limit[0], x1_limit[1])
                plot_axis.set_ylim(x2_limit[0], x2_limit[1])
                plot_axis.set_zlim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x1_label, fontsize=aer_config.font_size)
                plot_axis.set_ylabel(x2_label, fontsize=aer_config.font_size)
                plot_axis.set_zlabel(y_label, fontsize=aer_config.font_size)

            # finalize performance plot
            plot_axis.set_title(key, fontsize=aer_config.font_size)

            plot_filepath = os.path.join(self.results_plots_path, 'plot_' + plot_label + '_' + key + '.png')
            plot_fig.savefig(plot_filepath)

    def clear_validation_sets(self):
        self._validation_sets = dict()

    def add_validation_set(self, object_of_study, name=None):
        if name is None:
            name = object_of_study.get_name() + "_" + str(len(self._validation_sets.keys()))

        self._validation_sets[name] = object_of_study
        
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
    def init_model_search(self, object_of_study):
        pass

    @abstractmethod
    def run_model_search_epoch(self, object_of_study):
        pass

    @abstractmethod
    def log_model_search(self, object_of_study):
        pass

    @abstractmethod
    def init_meta_evaluation(self, object_of_study):
        pass

    @abstractmethod
    def init_model_evaluation(self, object_of_study):
        pass

    @abstractmethod
    def run_eval_epoch(self, object_of_study):
        pass

    @abstractmethod
    def log_model_evaluation(self, object_of_study):
        pass

    @abstractmethod
    def log_meta_evaluation(self, object_of_study):
        pass

    @abstractmethod
    def get_best_model(self, object_of_study):
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

    @abstractmethod
    def _meta_parameters_to_str(self):
        pass

    @abstractmethod
    def _eval_meta_parameters_to_str(self):
        pass

    def setup_simulation_directories(self):

        # study directory
        self.study_path = aer_config.studies_folder \
                          + self.study_name + "/"

        if not os.path.exists(self.study_path):
            os.mkdir(self.study_path)

        # scripts directory
        self.scripts_path = aer_config.studies_folder\
                            + self.study_name + "/"\
                            + aer_config.models_folder\
                            + aer_config.models_scripts_folder

        if not os.path.exists(self.scripts_path):
            os.mkdir(self.scripts_path)

        # results directory
        self.results_path = aer_config.studies_folder\
                            + self.study_name + "/"\
                            + aer_config.models_folder \
                            + aer_config.models_results_folder

        self.results_plots_path = aer_config.studies_folder\
                            + self.study_name + "/"\
                            + aer_config.models_folder \
                            + aer_config.models_results_plots_folder

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

        if not os.path.exists(self.results_plots_path):
            os.mkdir(self.results_plots_path)

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
        plot_dict[aer_config.plot_key_type] = type
        plot_dict[aer_config.plot_key_x_data] = x
        plot_dict[aer_config.plot_key_y_data] = y
        if x_limit is not None:
            plot_dict[aer_config.plot_key_x_limit] = x_limit
        if y_limit is not None:
            plot_dict[aer_config.plot_key_y_limit] = y_limit
        if x_label is not None:
            plot_dict[aer_config.plot_key_x_label] = x_label
        if y_label is not None:
            plot_dict[aer_config.plot_key_y_label] = y_label
        if legend is not None:
            plot_dict[aer_config.plot_key_legend] = legend
        if image is not None:
            plot_dict[aer_config.plot_key_image] = image
        if x_model is not None:
            plot_dict[aer_config.plot_key_x_model] = x_model
        if y_model is not None:
            plot_dict[aer_config.plot_key_y_model] = y_model
        if x_highlighted is not None:
            plot_dict[aer_config.plot_key_x_highlighted_data] = x_highlighted
        if y_highlighted is not None:
            plot_dict[aer_config.plot_key_y_highlighted_data] = y_highlighted

        return plot_dict