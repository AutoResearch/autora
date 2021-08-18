from abc import ABC
from AER_theorist.theorist_darts import Theorist_DARTS

import AER_config as aer_config
import AER_theorist.darts.darts_config as darts_cfg
import warnings
import logging
import pandas
import time
import os


class Theorist_Random_DARTS(Theorist_DARTS, ABC):

    def __init__(self, study_name, theorist_filter='darts'):
        super(Theorist_Random_DARTS, self).__init__(study_name)

        self.theorist_name = 'random_darts'
        self.model_search_epochs = 0
        self.load_runtimes(theorist_filter)


    def load_runtimes(self, theorist_filter):

        self.results_path

        # read in all csv files
        files = list()
        for file in os.listdir(self.results_path):
            if file.endswith(".csv"):
                if 'timestamps' not in file:
                    continue

                if theorist_filter is not None:
                    if theorist_filter not in file:
                        continue
                files.append(os.path.join(self.results_path, file))

        # collect all run times
        all_runtimes = dict()
        runtimes = dict()
        all_runtimes[aer_config.log_key_timestamp] = list()
        runtimes[aer_config.log_key_timestamp] = list()
        meta_param_names = self._meta_parameter_names_to_str_list()
        for name in meta_param_names:
            all_runtimes[name] = list()
            runtimes[name] = list()

        for file in files:
            data = pandas.read_csv(file, header=0)
            for name in meta_param_names:
                if name in data.keys():
                    all_runtimes[name].extend(data[name])
                else:
                    raise Exception('Could not find meta parameter' + name + '" in the data file: ' + str(file))

            if aer_config.log_key_timestamp in data.keys():
                all_runtimes[aer_config.log_key_timestamp].extend(data[aer_config.log_key_timestamp])
            else:
                raise Exception('Could not find timestamp key' + aer_config.log_key_timestamp + '" in the data file: ' + str(file))

        # for each meta-parameter pick highest run time
        decay_name = meta_param_names[0]
        k_name = meta_param_names[1]
        s_name = meta_param_names[2]

        for arch_weight_decay_df in darts_cfg.arch_weight_decay_list:
            for num_graph_nodes in darts_cfg.num_node_list:
                for seed in darts_cfg.seed_list:
                    max_time = 0
                    for idx in range(len(all_runtimes[aer_config.log_key_timestamp])):

                        decay = all_runtimes[decay_name][idx]
                        k = all_runtimes[k_name][idx]
                        s = all_runtimes[s_name][idx]

                        if decay == arch_weight_decay_df and num_graph_nodes == k and seed == s:
                            time_elapsed = all_runtimes[aer_config.log_key_timestamp][idx]
                            if time_elapsed > max_time:
                                max_time = time_elapsed

                    if max_time == 0:
                        warnings.warn("No time elapsed found for parameter configuration: decay=" + str(arch_weight_decay_df) + ", k=" + str(num_graph_nodes) + ", seed=" + str(seed))
                        max_time = 1

                    runtimes[decay_name].append(arch_weight_decay_df)
                    runtimes[k_name].append(num_graph_nodes)
                    runtimes[s_name].append(seed)
                    runtimes[aer_config.log_key_timestamp].append(max_time)

        self.runtimes = pandas.DataFrame.from_dict(runtimes)

    def update_runtime(self):
        # get meta parameters
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        names = self._meta_parameter_names_to_str_list()
        decay_name = names[0]
        num_graph_nodes_name = names[1]
        seed_name = names[2]

        # retrieve and save current runtime
        row = self.runtimes.loc[(self.runtimes[decay_name] == arch_weight_decay_df) & (self.runtimes[num_graph_nodes_name] == num_graph_nodes) & (self.runtimes[seed_name] == seed)]
        self.current_runtime = row[aer_config.log_key_timestamp].values[0]


    def init_model_search(self, object_of_study):
        super(Theorist_Random_DARTS, self).init_model_search(object_of_study)

        self.model._architecture_fixed = True
        self.model.alphas_normal.data[:] = 1
        self.update_runtime()

    def init_meta_evaluation(self, object_of_study=None):
        super(Theorist_Random_DARTS, self).init_meta_evaluation(object_of_study)

        # set up meta parameters for model evaluation
        self._eval_meta_parameters = list()
        self._eval_meta_parameters_iteration = 0
        for arch_sample in range(1):
            for init_sample in range(darts_cfg.n_initializations_sampled):
                meta_parameters = [arch_sample, init_sample]
                self._eval_meta_parameters.append(meta_parameters)

    # incorporate time spent
    def evaluate_model_search(self, object_of_study):

        # initialize model search
        self.init_meta_evaluation(object_of_study)

        while True:

            [arch_sample_id, param_sample_id] = self.get_eval_meta_parameters()

            # perform architecture search for different hyper-parameters
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

            # move to next meta parameter
            self._eval_meta_parameters_iteration += 1

            # check if reached end meta parameters explored
            if self._eval_meta_parameters_iteration == len(self._eval_meta_parameters):

                # check if exceeded runtime
                stop = time.time()
                elapsed = stop - self.start_search_timestamp
                if elapsed >= self.current_runtime:
                    break

                else: # if did not exceed runtime, add more meta parameters
                    for init_sample in range(darts_cfg.n_initializations_sampled):
                        meta_parameters = [arch_sample_id+1, init_sample]
                        self._eval_meta_parameters.append(meta_parameters)

        # sum up meta evaluation
        self.log_meta_evaluation(object_of_study)


    def run_model_search_epoch(self, epoch):
        pass

    def plot_model_eval(self, object_of_study):
        pass

    def log_model_search(self, object_of_study):
        pass