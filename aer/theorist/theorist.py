import os
import sys
import glob
import shutil
import logging
import time
import csv
from typing import Callable

from .. import config as AER_config
from ..utils import Plot_Types, do_nothing

from matplotlib.figure import Figure

from abc import ABC, abstractmethod


class Theorist(ABC):
    _loss_plot_name = "Training & Validation Loss"
    _pattern_plot_name = "Target vs. Predicted Pattern"
    simulation_files = ""

    def __init__(self, study_name):

        self._cfg = None

        self.study_name = ""
        self.study_path = ""
        self.scripts_path = ""
        self.results_path = ""
        self.results_plots_path = ""
        self.results_weights_path = ""
        self.theorist_name = "theorist"

        self._meta_parameters = list()
        self._meta_parameters_iteration = 0
        self._eval_meta_parameters_iteration = 0

        self.generate_plots = False

        self._model_search_parameters = dict()
        self._performance_plots = dict()
        self._supplementary_plots = dict()
        self._validation_sets = dict()
        self.plot_name_list = list()

        self.target_pattern = []
        self.prediction_pattern = []

        self.model_search_epochs = 100
        self.eval_epochs = 100

        self.model_search_id = 0
        self.study_name = study_name
        self.single_job = False
        self.setup_simulation_directories()
        self.copy_scripts(scripts_to_save=glob.glob(self.simulation_files))

    @abstractmethod
    def get_model_search_parameters(self):
        pass

    def set_model_search_parameter(self, key, str_value):
        if key in self._model_search_parameters.keys():
            if self._model_search_parameters[key][1] is True:
                self._model_search_parameters[key][0] = self._model_search_parameters[key][2](str_value)
                self.assign_model_search_parameters()
            else:
                raise Exception("Not allowed to modify model search parameter '" + str(
                    key) + "'. Dictionary self._model_search_parameters indicates that the parameter cannot be modified.")
        else:
            raise Exception("Key '" + str(key) + "' not in dictionary self._model_search_parameters")

    def search_model_job(self, object_of_study, job_id):

        self.single_job = True

        # initialize model search
        self.init_meta_search(object_of_study)

        self._meta_parameters_iteration = job_id

        # perform architecture search for different hyper-parameters
        self.start_search_timestamp = time.time()
        self.init_model_search(object_of_study)
        for epoch in range(self.model_search_epochs):
            self.run_model_search_epoch(epoch)
            if self.generate_plots:
                self.log_plot_data(epoch, object_of_study)
        self.log_model_search(object_of_study)
        if self.generate_plots:
            self.plot_model_search(object_of_study)
        self.evaluate_model_search(object_of_study)
        self.log_time_elapsed()

        self.log_meta_search(object_of_study)

    def search_model(self, object_of_study):
        # initialize model search
        self.init_meta_search(object_of_study)

        # perform architecture search for different hyper-parameters
        for meta_params in self._meta_parameters:
            self.start_search_timestamp = time.time()
            self.init_model_search(object_of_study)
            for epoch in range(self.model_search_epochs):
                self.run_model_search_epoch(epoch)
                if self.generate_plots:
                    self.log_plot_data(epoch, object_of_study)
            self.log_model_search(object_of_study)
            if self.generate_plots:
                self.plot_model_search(object_of_study)
            self.evaluate_model_search(object_of_study)
            self.log_time_elapsed()
            self._meta_parameters_iteration += 1

        self.log_meta_search(object_of_study)

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
            if self.generate_plots:
                self.plot_model_eval(object_of_study)

            # log model evaluation
            self.log_model_evaluation(object_of_study)
            self._eval_meta_parameters_iteration += 1

        # sum up meta evaluation
        self.log_meta_evaluation(object_of_study)

    def plot_model_search(self, object_of_study):
        plot_label = self.theorist_name + "_" + self._meta_parameters_to_str()

        # get performance plots
        all_performance_plots = self.get_performance_plots(object_of_study)
        performance_plots = self.select_plots(all_performance_plots)
        self.save_plots(performance_plots, plot_label)

        # supplementary plots
        all_supplementary_plots = self.get_supplementary_plots(object_of_study)
        supplementary_plots = self.select_plots(all_supplementary_plots)
        self.save_plots(supplementary_plots, plot_label)

    def plot_model_eval(self, object_of_study):
        plot_label = self.theorist_name + "_" + self._meta_parameters_to_str() + "_" + self._eval_meta_parameters_to_str()

        # performance plots
        all_performance_plots = self.get_performance_plots(object_of_study)
        performance_plots = self.select_plots(all_performance_plots)

        self.save_plots(performance_plots, plot_label)

    def select_plots(self, full_plot_dict):
        plot_dict = dict()
        if len(self.plot_name_list) == 0:
            plot_dict = full_plot_dict
        else:
            for plot_name in self.plot_name_list:
                if plot_name in full_plot_dict.keys():
                    plot_dict[plot_name] = full_plot_dict[plot_name]

        return plot_dict

    def plot(self, plot=True, plot_name_list=None):
        self.generate_plots = plot
        if plot_name_list is not None:
            self.plot_name_list = plot_name_list

    def save_plots(self, plot_list, plot_label):

        for key in plot_list.keys():
            plot_dict = plot_list[key]

            type = plot_dict[AER_config.plot_key_type]

            plot_fig = Figure(figsize=(7, 7), dpi=100)

            if type == Plot_Types.SURFACE_SCATTER:
                plot_axis = plot_fig.add_subplot(111, projection='3d')
            else:
                plot_axis = plot_fig.add_subplot(111)

            # plot_fig.subplots_adjust(bottom=0.2)
            # plot_fig.subplots_adjust(left=0.35)

            if type == Plot_Types.LINE:

                # get relevant data
                x_data = plot_dict[AER_config.plot_key_x_data]
                y_data = plot_dict[AER_config.plot_key_y_data]
                x_limit = plot_dict[AER_config.plot_key_x_limit]
                y_limit = plot_dict[AER_config.plot_key_y_limit]
                x_label = plot_dict[AER_config.plot_key_x_label]
                y_label = plot_dict[AER_config.plot_key_y_label]
                legend = plot_dict[AER_config.plot_key_legend]

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]  # remove previous lines
                plots = list()
                for idx, (x, y, leg) in enumerate(zip(x_data, y_data, legend)):
                    plots.append(plot_axis.plot(x, y, AER_config.plot_colors[idx], label=leg))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=AER_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=AER_config.font_size)

                plot_axis.legend(loc=2, fontsize="small")


            elif type == Plot_Types.IMAGE:

                # get relevant data
                image = plot_dict[AER_config.plot_key_image]
                x_data = plot_dict[AER_config.plot_key_x_data]
                y_data = plot_dict[AER_config.plot_key_y_data]
                x_label = plot_dict[AER_config.plot_key_x_label]
                y_label = plot_dict[AER_config.plot_key_y_label]

                # generate image
                plot_axis.cla()
                plot_axis.imshow(image, interpolation='nearest', aspect='auto')
                x = x_data
                y = y_data
                plot_axis.plot(x, y, color='red')

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=AER_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=AER_config.font_size)

            elif type == Plot_Types.LINE_SCATTER:

                # get relevant data
                x_data = plot_dict[AER_config.plot_key_x_data]
                y_data = plot_dict[AER_config.plot_key_y_data]
                x_model = plot_dict[AER_config.plot_key_x_model]
                y_model = plot_dict[AER_config.plot_key_y_model]
                x_limit = plot_dict[AER_config.plot_key_x_limit]
                y_limit = plot_dict[AER_config.plot_key_y_limit]
                x_label = plot_dict[AER_config.plot_key_x_label]
                y_label = plot_dict[AER_config.plot_key_y_label]
                legend = plot_dict[AER_config.plot_key_legend]

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
                plot_axis.set_xlabel(x_label, fontsize=AER_config.font_size)
                plot_axis.set_ylabel(y_label, fontsize=AER_config.font_size)

                plot_axis.legend(loc=2, fontsize="small")

            elif type == Plot_Types.SURFACE_SCATTER:

                # get relevant data
                (x1_data, x2_data) = plot_dict[AER_config.plot_key_x_data]
                y_data = plot_dict[AER_config.plot_key_y_data]
                (x1_model, x2_model) = plot_dict[AER_config.plot_key_x_model]
                y_model = plot_dict[AER_config.plot_key_y_model]
                (x1_limit, x2_limit) = plot_dict[AER_config.plot_key_x_limit]
                y_limit = plot_dict[AER_config.plot_key_y_limit]
                (x1_label, x2_label) = plot_dict[AER_config.plot_key_x_label]
                y_label = plot_dict[AER_config.plot_key_y_label]
                legend = plot_dict[AER_config.plot_key_legend]

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
                plot_axis.set_xlabel(x1_label, fontsize=AER_config.font_size)
                plot_axis.set_ylabel(x2_label, fontsize=AER_config.font_size)
                plot_axis.set_zlabel(y_label, fontsize=AER_config.font_size)

            # finalize performance plot
            plot_axis.set_title(key, fontsize=AER_config.font_size)

            plot_filepath = os.path.join(self.results_plots_path, 'plot_' + plot_label + '_' + key + '.png')
            plot_fig.savefig(plot_filepath)

    def log_time_elapsed(self):
        stop = time.time()
        elapsed = stop - self.start_search_timestamp

        meta_param_names = self._meta_parameter_names_to_str_list()
        meta_param_values = self._meta_parameter_values_to_str_list()

        for name, value in zip(meta_param_names, meta_param_values):
            self.time_elapsed_log[name].append(value)
        self.time_elapsed_log[AER_config.log_key_timestamp].append(str(elapsed))

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

        # initialize dictionary for logging the time elapsed associated with each meta search condition
        self.time_elapsed_log = dict()
        names = self._meta_parameter_names_to_str_list()
        for name in names:
            self.time_elapsed_log[name] = list()
        self.time_elapsed_log[AER_config.log_key_timestamp] = list()

    InterruptCallable = Callable[[], bool]
    EventCallable = Callable[[str, dict], None]

    @abstractmethod
    def run_meta_search(self,
                        object_of_study,
                        resume=False,
                        last_meta_param_idx=0,
                        last_epoch=0,

                        # Callbacks which can interrupt the program
                        check_paused: InterruptCallable = lambda: False,
                        check_running: InterruptCallable = lambda: True,
                        check_not_running: InterruptCallable = lambda: False,

                        # Callbacks which access local variables
                        event_handler: EventCallable = do_nothing,
                        ):
        """ Run architecture search for different hyper-parameters.

        This method offers a series of callbacks, which are of two types:
         - InterruptCallable: can be used to interrupt (pause, stop) the execution
             - Default: allow the program to continue
             - Implementation: Callable of no arguments which returns a boolean.

         - AccessLocalsCallable: can be used to access local variables from outside the evaluation.
            These can be used for updating external interfaces, like a CLI, GUI, message queue,
            or to write current results to file.
             - Default: do nothing.
             - These are called within the method like foo_callback(**locals())
             - Implementation: Callable must accept all locals() from the run_meta_search.
                Usually defined with a signature like follows: foo_callback(local_1, local_2, **kwargs)

        """
        def emit_event(label: str, data: dict):
            event_handler(label, data)

        model_search_epochs = self.model_search_epochs
        num_meta_idx = len(self._meta_parameters)
        num_eval_epochs = self.eval_epochs

        for idx, meta_params in enumerate(self._meta_parameters):

            if resume is True:
                if idx < last_meta_param_idx:
                    continue
            else:

                [arch_weight_decay_df, num_graph_nodes, seed] = meta_params
                self.init_model_search(object_of_study)

                meta_param_str = self._meta_parameters_to_str()

                model_search_parameters = self.get_model_search_parameters()
                performance_plots = self.get_performance_plots(object_of_study)
                supplementary_plots = self.get_supplementary_plots(object_of_study)

                emit_event('post-meta-search', locals())

            for epoch in range(model_search_epochs):

                if resume is True:
                    if epoch < last_epoch:
                        continue

                # check if still running
                if check_not_running():
                    break

                # check if paused
                if check_paused():
                    emit_event('paused-model-search', locals())
                    return

                # update run button
                emit_event('pre-model-search', locals())


                #### Key function call
                self.run_model_search_epoch(epoch)
                ####
                # update parameter list
                # Log plot data
                model_search_parameters = self.get_model_search_parameters()
                self.log_plot_data(epoch, object_of_study)
                # update performance plot
                performance_plots = self.get_performance_plots(object_of_study)
                # update supplementary plot
                supplementary_plots = self.get_supplementary_plots(object_of_study)
                emit_event('post-model-search', locals())

            if check_running():
                self.log_model_search(object_of_study)

                # save all performance plots
                performance_plots = self.get_performance_plots(object_of_study)

                # save all supplementary plots
                supplementary_plots = self.get_supplementary_plots(object_of_study)

                emit_event('pre-meta-evaluation', locals())

                # Theorist: evaluate model architecture

                # initialize meta evaluation
                self.init_meta_evaluation(object_of_study)

                # perform architecture search for different hyper-parameters
                for eval_meta_params in self._eval_meta_parameters:

                    eval_param_str = self._eval_meta_parameters_to_str()

                    emit_event('pre-model-eval', locals())

                    self.init_model_evaluation(object_of_study)

                    # loop over epochs
                    for epoch in range(self.eval_epochs):
                        # run single epoch
                        self.run_eval_epoch(epoch, object_of_study)
                        # log performance (for plotting purposes)
                        self.log_plot_data(epoch, object_of_study)

                        performance_plots = self.get_performance_plots(object_of_study)
                        emit_event('post-model-eval-epoch', locals())

                    # save all performance plots
                    performance_plots = self.get_performance_plots(object_of_study)
                    emit_event('post-model-eval', locals())

                    # log model evaluation
                    self.log_model_evaluation(object_of_study)

                # sum up meta evaluation
                self.log_meta_evaluation(object_of_study)

                self._meta_parameters_iteration += 1

        best_model = self.get_best_model(object_of_study, plot_model=True)

        return best_model

    @abstractmethod
    def log_meta_search(self, object_of_study):
        # write time elapsed for each meta parameter condition to csv
        self.time_elapsed_to_csv()

    def time_elapsed_to_csv(self):

        # timestamps to csv
        if self.single_job:
            meta_param_str = '_' + self._meta_parameters_to_str()
        else:
            meta_param_str = ''
        filename_csv = self.theorist_name + meta_param_str + '_search_' + str(self.model_search_id) + '_timestamps.csv'
        filepath = os.path.join(self.results_path, filename_csv)

        # format data
        header = list()
        zip_data = list()
        # add log data from additional validation sets
        for key in self.time_elapsed_log.keys():
            header.append(key)
            timestamp_log = self.time_elapsed_log[key]
            zip_data.append(timestamp_log)

        rows = zip(*zip_data)

        # write data to csv
        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

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
    def _meta_parameter_names_to_str_list(self):
        pass

    @abstractmethod
    def _meta_parameter_values_to_str_list(self):
        pass

    @abstractmethod
    def _eval_meta_parameters_to_str(self):
        pass

    def setup_simulation_directories(self):

        # study directory
        self.study_path = AER_config.studies_folder \
                          + self.study_name + "/"

        if not os.path.exists(self.study_path):
            os.mkdir(self.study_path)

        # scripts directory
        self.scripts_path = AER_config.studies_folder \
                            + self.study_name + "/" \
                            + AER_config.models_folder \
                            + AER_config.models_scripts_folder

        if not os.path.exists(self.scripts_path):
            os.mkdir(self.scripts_path)

        # results directory
        self.results_path = AER_config.studies_folder \
                            + self.study_name + "/" \
                            + AER_config.models_folder \
                            + AER_config.models_results_folder

        self.results_plots_path = AER_config.studies_folder \
                                  + self.study_name + "/" \
                                  + AER_config.models_folder \
                                  + AER_config.models_results_plots_folder

        self.results_weights_path = AER_config.studies_folder \
                                    + self.study_name + "/" \
                                    + AER_config.models_folder \
                                    + AER_config.models_results_weights_folder

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

        if not os.path.exists(self.results_plots_path):
            os.mkdir(self.results_plots_path)

        if not os.path.exists(self.results_weights_path):
            os.mkdir(self.results_weights_path)

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

    def _generate_plot_dict(self, type, x, y, x_limit=None, y_limit=None, x_label=None, y_label=None, legend=None,
                            image=None, x_model=None, y_model=None, x_highlighted=None, y_highlighted=None):
        # generate plot dictionary
        plot_dict = dict()
        plot_dict[AER_config.plot_key_type] = type
        plot_dict[AER_config.plot_key_x_data] = x
        plot_dict[AER_config.plot_key_y_data] = y
        if x_limit is not None:
            plot_dict[AER_config.plot_key_x_limit] = x_limit
        if y_limit is not None:
            plot_dict[AER_config.plot_key_y_limit] = y_limit
        if x_label is not None:
            plot_dict[AER_config.plot_key_x_label] = x_label
        if y_label is not None:
            plot_dict[AER_config.plot_key_y_label] = y_label
        if legend is not None:
            plot_dict[AER_config.plot_key_legend] = legend
        if image is not None:
            plot_dict[AER_config.plot_key_image] = image
        if x_model is not None:
            plot_dict[AER_config.plot_key_x_model] = x_model
        if y_model is not None:
            plot_dict[AER_config.plot_key_y_model] = y_model
        if x_highlighted is not None:
            plot_dict[AER_config.plot_key_x_highlighted_data] = x_highlighted
        if y_highlighted is not None:
            plot_dict[AER_config.plot_key_y_highlighted_data] = y_highlighted

        return plot_dict
