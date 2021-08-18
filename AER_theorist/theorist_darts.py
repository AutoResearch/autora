from abc import ABC
from AER_theorist.theorist import Theorist
from AER_utils import Plot_Types
from AER_theorist.darts.model_search import Network, DARTS_Type
from AER_theorist.darts.architect import Architect
from AER_theorist.darts.genotypes import PRIMITIVES
from torch.autograd import Variable
from AER_experimentalist.experiment_environment.variable import outputTypes as output_types

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
import logging
import numpy as np
import AER_theorist.darts.darts_config as darts_cfg
import AER_config as aer_config
import AER_theorist.darts.utils as utils
import AER_theorist.darts.visualize as viz
import copy
import os
import csv


class Theorist_DARTS(Theorist, ABC):

    simulation_files = 'AER_theorist/darts/*.py'

    _lr_plot_name = "Learning Rates"

    def __init__(self, study_name, darts_type=DARTS_Type.ORIGINAL):
        super(Theorist_DARTS, self).__init__(study_name)

        self.DARTS_type = darts_type
        self.criterion = None

        self._model_summary_list = list()
        self._eval_meta_parameters = list()  # meta parameters for model evaluation
        self._meta_parameters = list()  # meta parameters for model search

        self._eval_criterion_loss_log = list()
        self._eval_model_name_log = list()
        self._eval_arch_name_log = list()
        self._eval_num_graph_node_log = list()
        self._eval_theorist_log = list()
        self._eval_arch_weight_decay_log = list()
        self._eval_num_params_log = list()
        self._eval_num_edges_log = list()

        self.model_search_epochs = darts_cfg.epochs
        self.eval_epochs = darts_cfg.eval_epochs

        if self.DARTS_type == DARTS_Type.ORIGINAL:
            self.theorist_name = 'original_darts'
        elif self.DARTS_type == DARTS_Type.FAIR:
            self.theorist_name = 'fair_darts'
        else:
            raise Exception("DARTS Type " + str(self.DARTS_type) + " not implemented")


    def get_model_search_parameters(self):

        lm_float = lambda x: float(x)
        lm_int = lambda x: int(x)
        lm_bool = lambda x: bool(x)

        # architecture parameters
        self._model_search_parameters["arch weight decay"] = [self.architect.optimizer.param_groups[0]['weight_decay'], True, lm_float]
        self._model_search_parameters["arch lr"] = [self.architect.optimizer.param_groups[0]['lr'], True, lm_float]
        self._model_search_parameters["arch weight decay df"] = [self.architect.network_weight_decay_df, True, lm_float] # requires call to architect._init_decay_weights()
        self._model_search_parameters["arch unrolled"] = [darts_cfg.unrolled, True, lm_bool]

        # network parameters
        self._model_search_parameters["params momentum"] = [self.optimizer.param_groups[0]['momentum'], True, lm_float]
        self._model_search_parameters["params weight decay"] = [self.optimizer.param_groups[0]['weight_decay'], True, lm_float]
        self._model_search_parameters["classifier weight decay"] = [self.model._classifier_weight_decay, True, lm_float]
        self._model_search_parameters["params max lr"] = [darts_cfg.learning_rate, False, lm_float] # self.optimizer.param_groups[0]['lr']
        self._model_search_parameters["params min lr"] = [darts_cfg.learning_rate_min, False, lm_float]

        # training protocol parameters
        self._model_search_parameters["training set proportion"] = [darts_cfg.train_portion, False, lm_float]
        self._model_search_parameters["batch size"] = [darts_cfg.batch_size, False, lm_int]

        return self._model_search_parameters

    def assign_model_search_parameters(self):

        # architecture parameters
        self.architect.optimizer.param_groups[0]['weight_decay'] = self._model_search_parameters["arch weight decay"][0]
        self.architect.optimizer.param_groups[0]['lr'] = self._model_search_parameters["arch lr"][0]
        self.architect.network_weight_decay_df = self._model_search_parameters["arch weight decay df"][0]
        self.architect._init_decay_weights()
        darts_cfg.unrolled = self._model_search_parameters["arch unrolled"][0]

        # network parameters
        self.optimizer.param_groups[0]['momentum'] = self._model_search_parameters["params momentum"][0]
        self.optimizer.param_groups[0]['weight_decay'] = self._model_search_parameters["params weight decay"][0]
        self.model._classifier_weight_decay = self._model_search_parameters["classifier weight decay"][0]
        self.optimizer.param_groups[0]['lr'] = self._model_search_parameters["params current lr"][0]
        # self.scheduler.eta_min = self._model_search_parameters["params min lr"][0]

    def init_meta_search(self, object_of_study):
        super(Theorist_DARTS, self).init_meta_search(object_of_study)

        # clear model summary list
        self._model_summary_list = list()

        # define loss function
        self.criterion = utils.get_loss_function(object_of_study.__get_output_type__())

        # set configuration
        self._cfg = darts_cfg

        # log: gpu device, parameter configuration
        logging.info('gpu device: %d' % darts_cfg.gpu)
        logging.info("configuration = %s", darts_cfg)

        # sets seeds
        np.random.seed(int(darts_cfg.seed))
        torch.manual_seed(int(darts_cfg.seed))

        # set up meta parameters
        self._meta_parameters = list()
        self._meta_parameters_iteration = 0
        for arch_weight_decay_df in darts_cfg.arch_weight_decay_list:
            for num_graph_nodes in darts_cfg.num_node_list:
                for seed in darts_cfg.seed_list:
                    meta_parameters = [arch_weight_decay_df, int(num_graph_nodes), seed]
                    self._meta_parameters.append(meta_parameters)

    def get_meta_parameters(self, iteration = None):
        if iteration is None:
            iteration = self._meta_parameters_iteration

        return self._meta_parameters[iteration]

    def get_eval_meta_parameters(self, iteration = None):
        if iteration is None:
            iteration = self._eval_meta_parameters_iteration

        return self._eval_meta_parameters[iteration]

    def get_next_meta_parameters(self):
        self._meta_parameters_iteration += 1
        return self.get_meta_parameters()

    def commission_meta_search(self, object_of_study):
        raise Exception('Not implemented.')
        pass

    def get_best_model(self, object_of_study, plot_model=False):

        # determine best model
        best_loss = None
        best_model_file = None
        best_arch_file = None
        for summary_file in self._model_summary_list:
            # read CSV
            data = pandas.read_csv(summary_file, header=0)

            log_losses = np.asarray(data[darts_cfg.csv_loss])
            log_losses = log_losses.astype(float)
            min_loss_index = np.argmin(log_losses)

            best_local_log_loss = log_losses[min_loss_index]
            if best_loss is None or best_local_log_loss < best_loss:
                best_loss = best_local_log_loss
                best_model_file = data[darts_cfg.csv_model_file_name][min_loss_index]
                best_arch_file = data[darts_cfg.csv_arch_file_name][min_loss_index]
                best_num_graph_nodes = int(data[darts_cfg.csv_num_graph_node][min_loss_index])

        # load winning model
        model_path = os.path.join(self.results_weights_path, best_model_file + ".pt")
        arch_path = os.path.join(self.results_weights_path, best_arch_file + ".pt")
        model = Network(object_of_study.__get_output_dim__(),
                        self.criterion,
                        steps=best_num_graph_nodes,
                        n_input_states=object_of_study.__get_input_dim__(),
                        darts_type=self.DARTS_type)
        utils.load(model, model_path)
        alphas_normal = torch.load(arch_path)
        model.fix_architecture(True, new_weights=alphas_normal)

        # return winning model
        self.model = model

        # plot model
        if plot_model:
            filename = "best_model_" + self.theorist_name
            best_model_plot_path = os.path.join(self.results_path, filename)
            genotype = self.model.genotype()
            (n_params_total, n_params_base, param_list) = self.model.countParameters()
            viz.plot(genotype.normal, best_model_plot_path, fileFormat='png',
                     input_labels=object_of_study.__get_input_labels__(), full_label=True, param_list=param_list,
                     out_dim=object_of_study.__get_output_dim__(), out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))

        return model

    # def evaluate_model_search(self, object_of_study):
    #
    #     [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
    #     model_eval_filepath = self.evaluate_architectures(object_of_study, self.train_queue, self.valid_queue,
    #                                                       self.model,
    #                                                       arch_weight_decay_df, num_graph_nodes, seed)
    #     self._model_summary_list.append(model_eval_filepath)

    def init_meta_evaluation(self, object_of_study=None):

        model = self.model
        max_num_architectures = model.max_alphas_normal().shape[0] * model.max_alphas_normal().shape[1]
        n_architectures_sampled = np.min([max_num_architectures, darts_cfg.n_architectures_sampled])

        # set up meta parameters for model evaluation
        self._eval_meta_parameters = list()
        self._eval_meta_parameters_iteration = 0
        for arch_sample in range(n_architectures_sampled):
            for init_sample in range(darts_cfg.n_initializations_sampled):
                meta_parameters = [arch_sample, init_sample]
                self._eval_meta_parameters.append(meta_parameters)

        # set up log components
        self._eval_criterion_loss_log = list()
        self._eval_model_name_log = list()
        self._eval_arch_name_log = list()
        self._eval_num_graph_node_log = list()
        self._eval_theorist_log = list()
        self._eval_arch_weight_decay_log = list()
        self._eval_num_params_log = list()
        self._eval_num_edges_log = list()
        self._validation_log_list = dict()
        for key in self._validation_sets:
            self._validation_log_list[key] = list()

        self.candidate_weights = None
        self.current_arch_sample_id = None

        # generate general model file name
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        self._eval_model_filename_gen = self.get_model_weights_filename(arch_weight_decay_df, num_graph_nodes, seed)
        self._eval_summary_filename_gen = self.get_model_filename(arch_weight_decay_df, num_graph_nodes, seed)
        self._eval_arch_filename_gen = self.get_architecture_filename(arch_weight_decay_df, num_graph_nodes, seed)

        # subsample models and retrain
        self._eval_sampled_weights = list()

        if self.DARTS_type == DARTS_Type.ORIGINAL:
            self.sample_amp = darts_cfg.sample_amp
        elif self.DARTS_type == DARTS_Type.FAIR:
            self.sample_amp = darts_cfg.sample_amp_fair_darts


    def init_model_evaluation(self, object_of_study):

        self.train_error_log = np.empty((self.eval_epochs, 1))  # log training error
        self.valid_error_log = np.empty((self.eval_epochs, 1))  # log validation err
        self.param_lr_log = np.empty((self.eval_epochs, 1))  # log model learning rate
        self.arch_lr_log = np.empty((self.eval_epochs, 1))  # log architecture learning rate
        self.architecture_weights_log = np.empty(
            (self.eval_epochs, self.num_arch_edges, self.num_arch_ops))  # log architecture weights
        self.train_error_log[:] = np.nan
        self.valid_error_log[:] = np.nan
        self.param_lr_log[:] = np.nan
        self.arch_lr_log[:] = np.nan
        self.architecture_weights_log[:] = np.nan

        # get model search meta parameters
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()

        # get evaluation meta parameters (architecture id and parameterization id)
        [arch_sample_id, param_sample_id] = self.get_eval_meta_parameters()
        n_eval_meta_configurations = len(self._eval_meta_parameters)

        logging.info('architecture evaluation for sampled model: %d / %d', self._eval_meta_parameters_iteration + 1, n_eval_meta_configurations)

        if self.current_arch_sample_id is None:
            sample_new_weights = True
        else:
            # if this is the same arch sample id then do not sample new weights
            if arch_sample_id == self.current_arch_sample_id:
                sample_new_weights = False
            else:
                sample_new_weights = True

        if sample_new_weights:
            # sample architecture weights
            found_weights = False
            if (arch_sample_id == 0):
                candidate_weights = self.model.max_alphas_normal()
                found_weights = True
            else:
                candidate_weights = self.model.sample_alphas_normal(sample_amp=self.sample_amp,
                                                                    fair_darts_weight_threshold=darts_cfg.fair_darts_weight_threshold)

            arch_search_attempt = 0
            while found_weights is False:
                weights_are_novel = True
                for logged_weights in self._eval_sampled_weights:
                    if torch.eq(logged_weights, candidate_weights).all():
                        weights_are_novel = False
                if weights_are_novel:
                    novel_weights = candidate_weights
                    found_weights = True
                else:
                    candidate_weights = self.model.sample_alphas_normal(sample_amp=self.sample_amp,
                                                                        fair_darts_weight_threshold=darts_cfg.fair_darts_weight_threshold)
                    if arch_search_attempt > darts_cfg.max_arch_search_attempts:
                        found_weights = True
                arch_search_attempt += 1

        else:
            # use old weights
            candidate_weights = self.candidate_weights

        # store sampled architecture weights
        self._eval_sampled_weights.append(candidate_weights)
        self.candidate_weights = candidate_weights
        self.current_arch_sample_id = arch_sample_id

        # sample parameter initialization

        # reinitialize weights if desired
        if darts_cfg.reinitialize_weights:
            self._eval_model = Network(object_of_study.__get_output_dim__(), self.criterion, steps=int(num_graph_nodes),
                                n_input_states=object_of_study.__get_input_dim__(),
                                classifier_weight_decay=darts_cfg.classifier_weight_decay,
                                darts_type=self.DARTS_type)
            if darts_cfg.eval_custom_initialization:
                self._eval_model.apply(init_weights)
        else:
            self._eval_model = copy.deepcopy(self.model)

        self._eval_model.fix_architecture(True, candidate_weights)

        # optimizer is standard stochastic gradient decent with some momentum and weight decay
        self._eval_optimizer = torch.optim.SGD(
            self._eval_model.parameters(),
            darts_cfg.eval_learning_rate,
            momentum=darts_cfg.eval_momentum,
            weight_decay=darts_cfg.eval_weight_decay)

        # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
        self._eval_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._eval_optimizer, float(darts_cfg.eval_epochs), eta_min=darts_cfg.eval_learning_rate_min)

        self._eval_model.train()  # Sets the module in training mode

        # set model to currently evaluated model (for plot purposes)
        self._eval_loss_log = list()
        self._model_org = copy.deepcopy(self.model)
        self.model = self._eval_model

    def run_eval_epoch(self, epoch, object_of_study):

        # get new learning rate
        lr = self._eval_scheduler.get_last_lr()[0]

        # get input and target
        input_search, target_search = next(iter(self.train_queue))
        input = Variable(input_search, requires_grad=False)  # .cuda()
        target = Variable(target_search, requires_grad=False)  # .cuda(async=True)

        input, target = format_input_target(input, target, self.criterion)

        # zero out gradients
        self._eval_optimizer.zero_grad()
        # compute loss for the model
        logits = self._eval_model(input)
        loss = self.criterion(logits, target)
        # update gradients for model
        loss.backward()
        # clips the gradient norm
        nn.utils.clip_grad_norm_(self._eval_model.parameters(), darts_cfg.grad_clip)
        # moves optimizer one step (applies gradients to weights)
        self._eval_optimizer.step()
        # applies weight decay to classifier weights
        self._eval_model.apply_weight_decay_to_classifier(lr)

        # if in debug mode, print loss during architecture evaluation

        # training loss
        logging.info('criterion loss %f', loss)
        self._eval_loss_log.append(loss.detach().numpy())
        # moves the annealing scheduler forward to determine new learning rate
        self._eval_scheduler.step()

        # validation loss
        validation_loss = infer(self.valid_queue, self._eval_model, self.criterion, silent=True)

        self.train_error_log[epoch] = loss.detach().numpy()
        self.valid_error_log[epoch] = validation_loss.detach().numpy()

    def _validate_model(self, key):
        if key in self._validation_sets.keys():
            # retrieve object of study
            object_of_study = self._validation_sets[key]
            (input, target) = object_of_study.get_dataset()

            # determine criterion
            output_type = object_of_study.__get_output_type__()
            criterion = utils.get_loss_function(output_type)

            # compute loss
            logits = self._eval_model(input)

            # compute BIC appropriate output types
            if output_type == output_types.CLASS or output_type == output_types.PROBABILITY_SAMPLE:
                loss = utils.compute_BIC(output_type, self._eval_model, input, target)
            else:
                if output_type == output_types.CLASS:
                    loss = criterion(logits, torch.flatten(target.long()))
                else:
                    loss = criterion(logits, target)

            loss = loss.detach().numpy()
            return loss

        else:
            raise Exception('No validation set named "' + key + '".')

    def log_model_evaluation(self, object_of_study):

        # get meta parameters
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        [arch_sample_id, param_sample_id] = self.get_eval_meta_parameters()

        # evaluate model
        criterion_loss = infer(self.valid_queue, self._eval_model, self.criterion, silent=True)
        self._eval_criterion_loss_log.append(criterion_loss.numpy())

        # for each validation set, compute validation log
        for idx, key in enumerate(self._validation_sets.keys()):
            loss = self._validate_model(key)
            self._validation_log_list[key].append(loss)

        # get model name
        model_filename = self._eval_model_filename_gen + '_sample' + str(arch_sample_id) + '_' + str(param_sample_id)
        arch_filename = self._eval_arch_filename_gen + '_sample' + str(arch_sample_id) + '_' + str(param_sample_id)
        model_filepath = os.path.join(self.results_weights_path, model_filename + '.pt')
        arch_filepath = os.path.join(self.results_weights_path, arch_filename + '.pt')
        model_graph_filepath = os.path.join(self.results_path, model_filename)
        self._eval_model_name_log.append(model_filename)
        self._eval_arch_name_log.append(arch_filename)
        self._eval_num_graph_node_log.append(num_graph_nodes)
        self._eval_arch_weight_decay_log.append(arch_weight_decay_df)
        num_params, _, _ = self._eval_model.countParameters()
        self._eval_num_params_log.append(num_params)
        num_non_zero_edges = self._eval_model.alphas_normal.data.shape[0] - int(np.sum(self._eval_model.alphas_normal.data[:,PRIMITIVES.index('none')].numpy()))
        self._eval_num_edges_log.append(num_non_zero_edges)
        genotype = self._eval_model.genotype()
        self._eval_theorist_log.append(self.theorist_name)


        # save model
        utils.save(self._eval_model, model_filepath)
        torch.save(self._eval_model.alphas_normal, arch_filepath)
        print('Saving model weights: ' + model_filepath)
        (n_params_total, n_params_base, param_list) = self._eval_model.countParameters()
        viz.plot(genotype.normal, model_graph_filepath, viewFile=False,
                 input_labels=object_of_study.__get_input_labels__(), param_list=param_list, full_label=True,
                 out_dim=object_of_study.__get_output_dim__(), out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))
        print('Saving model graph: ' + model_graph_filepath)
        print('Saving architecture weights: ' + arch_filepath)

        self.model = self._model_org

    def log_meta_evaluation(self, object_of_study):

        # get name for csv log file
        model_filename_csv = self._eval_summary_filename_gen + '.csv'
        model_filepath = os.path.join(self.results_path, model_filename_csv)

        # save csv file

        # generate header
        header = [darts_cfg.csv_theorist_name, darts_cfg.csv_model_file_name, darts_cfg.csv_arch_file_name, darts_cfg.csv_num_graph_node,
                  darts_cfg.csv_arch_weight_decay, darts_cfg.csv_num_params, darts_cfg.csv_num_edges,
                  darts_cfg.csv_loss]

        # collect log data
        zip_data = list()
        zip_data.append(self._eval_theorist_log)
        zip_data.append(self._eval_model_name_log)
        zip_data.append(self._eval_arch_name_log)
        zip_data.append(self._eval_num_graph_node_log)
        zip_data.append(self._eval_arch_weight_decay_log)
        zip_data.append(self._eval_num_params_log)
        zip_data.append(self._eval_num_edges_log)
        zip_data.append(self._eval_criterion_loss_log)

        # add log data from additional validation sets
        for key in self._validation_log_list.keys():
            header.append(key)
            validation_log = self._validation_log_list[key]
            zip_data.append(validation_log)

        rows = zip(*zip_data)

        with open(model_filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

        self._model_summary_list.append(model_filepath)

    def init_model_search(self, object_of_study):

        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()

        # initializes the model given number of channels, output classes and the training criterion
        self.model = Network(object_of_study.__get_output_dim__(), self.criterion, steps=int(num_graph_nodes),
                        n_input_states=object_of_study.__get_input_dim__(),
                        classifier_weight_decay=darts_cfg.classifier_weight_decay,
                        darts_type=self.DARTS_type)

        # initialize model
        if darts_cfg.custom_initialization:
            self.model.apply(init_weights)

        # log size of parameter space
        logging.info("param size: %fMB", utils.count_parameters_in_MB(self.model))

        # optimizer is standard stochastic gradient decent with some momentum and weight decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            darts_cfg.learning_rate,
            momentum=darts_cfg.momentum,
            weight_decay=darts_cfg.weight_decay)

        # determine training set
        train_data = object_of_study
        num_train = len(train_data)  # number of patterns
        indices = list(range(num_train))  # indices of all patterns
        split = int(np.floor(darts_cfg.train_portion * num_train))  # size of training set

        # combine the training set with a sampler, and provides an iterable over the training set
        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=darts_cfg.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        # combine the validation set with a sampler, and provides an iterable over the validation set
        self.valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=darts_cfg.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0)

        # generate an architecture of the model
        darts_cfg.arch_weight_decay_df = arch_weight_decay_df
        self.architect = Architect(self.model, darts_cfg)

        # plot variables
        self.num_arch_edges = self.model.alphas_normal.data.shape[0]  # number of architecture edges
        self.num_arch_ops = self.model.alphas_normal.data.shape[1]  # number of operations
        self.arch_ops_labels = PRIMITIVES  # operations
        self.train_error_log = np.empty((self.model_search_epochs, 1))  # log training error
        self.valid_error_log = np.empty((self.model_search_epochs, 1))  # log validation error
        self.param_lr_log = np.empty((self.model_search_epochs, 1))  # log model learning rate
        self.arch_lr_log = np.empty((self.model_search_epochs, 1))  # log architecture learning rate
        self.train_error_log[:] = np.nan
        self.valid_error_log[:] = np.nan
        self.param_lr_log[:] = np.nan
        self.arch_lr_log[:] = np.nan
        self.architecture_weights_log = np.empty(
            (self.model_search_epochs, self.num_arch_edges, self.num_arch_ops))  # log architecture weights
        self.architecture_weights_log[:] = np.nan

        graph_filename = utils.create_output_file_name(file_prefix=darts_cfg.graph_filename,
                                                       theorist=self.theorist_name,
                                                       log_version=self.model_search_id,
                                                       weight_decay=arch_weight_decay_df,
                                                       k=num_graph_nodes,
                                                       seed=seed)
        self.graph_filepath = os.path.join(self.results_path, graph_filename)

    def log_meta_search(self, object_of_study):
        super(Theorist_DARTS, self).log_meta_search(object_of_study)

    def run_model_search_epoch(self, epoch):

        # returns the genotype of the model
        genotype = self.model.genotype()
        # logs the genotype of the model
        logging.info('genotype: %s', genotype)

        # prints and log weights of the normal and reduced architecture
        if self.DARTS_type == DARTS_Type.ORIGINAL:
            print(F.softmax(self.model.alphas_normal, dim=-1))
        elif self.DARTS_type == DARTS_Type.FAIR:
            print(torch.sigmoid(self.model.alphas_normal))

        # training (for one epoch)
        train_obj = train(self.train_queue, self.valid_queue, self.model, self.architect, self.criterion, self.optimizer,
                          darts_cfg.arch_updates_per_epoch, darts_cfg.param_updates_per_epoch)
        # log accuracy on training set
        logging.info('training accuracy: %f', train_obj)

        # validation (for current epoch)
        valid_obj = infer(self.valid_queue, self.model, self.criterion)
        # log accuracy on validation set
        logging.info('validation accuracy: %f', valid_obj)

        logging.info('epoch: %d', epoch)

        self.train_error_log[epoch] = train_obj
        self.valid_error_log[epoch] = valid_obj

    def log_plot_data(self, epoch, object_of_study):
        self.param_lr_log[epoch] = self.optimizer.param_groups[0]['lr']
        self.arch_lr_log[epoch] = self.architect.optimizer.param_groups[0]['lr']

        # get full data set:
        (input, target) = object_of_study.get_dataset()
        self.target_pattern = target.detach().numpy()
        #self.prediction_pattern = self.model(input).detach().numpy()
        self.prediction_pattern = model_formatted(self.model, input, object_of_study).detach().numpy()

        # log architecture weights
        if self.DARTS_type == DARTS_Type.ORIGINAL:
            logged_weights = torch.nn.functional.softmax(self.model.alphas_normal, dim=-1).data.numpy()
        elif self.DARTS_type == DARTS_Type.FAIR:
            logged_weights = torch.sigmoid(self.model.alphas_normal).data.numpy()
        self.architecture_weights_log[epoch, :, :] = logged_weights


    def log_model_search(self, object_of_study):

        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        
        # save model plot
        genotype = self.model.genotype()
        viz.plot(genotype.normal, self.graph_filepath, fileFormat='png',
                 input_labels=object_of_study.__get_input_labels__(),
                 out_dim=object_of_study.__get_output_dim__(),
                 out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))

        # stores the model and architecture
        model_filename = self.get_model_weights_filename(arch_weight_decay_df, num_graph_nodes, seed)
        arch_filename = self.get_architecture_filename(arch_weight_decay_df, num_graph_nodes, seed)

        model_filepath = os.path.join(self.results_weights_path, model_filename + '.pt')
        arch_filepath = os.path.join(self.results_weights_path, arch_filename + '.pt')

        utils.save(self.model, model_filepath)
        torch.save(self.model.alphas_normal, arch_filepath)

    def plot_model(self, object_of_study, model=None, full_label=False):

        if model is None:
            model = self.model

        # get genotype
        genotype = model.genotype()

        if full_label:
            # get parameter list
            (n_params_total, n_params_base, param_list) = model.countParameters()

            # save model plot with parameters
            viz.plot(genotype.normal, self.graph_filepath, fileFormat='png',
                         input_labels=object_of_study.__get_input_labels__(), param_list=param_list, full_label=full_label,
                     out_dim=object_of_study.__get_output_dim__(), out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))

        else:
            # save model plot without parameters
            viz.plot(genotype.normal, self.graph_filepath, fileFormat='png',
                     input_labels=object_of_study.__get_input_labels__(),
                     out_dim=object_of_study.__get_output_dim__(), out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))

        return self.graph_filepath + ".png"

    def get_performance_plots(self, object_of_study):
        super(Theorist_DARTS, self).get_performance_plots(object_of_study)
        self.update_lr_plot()
        self.update_model_fit_plots(object_of_study)
        return self._performance_plots

    def get_supplementary_plots(self, object_of_study):
        self.update_arch_weights_plots()
        return self._supplementary_plots

    def update_lr_plot(self):

        if hasattr(self, 'param_lr_log') is not True:
            return

        # type
        type = Plot_Types.LINE

        # x data
        x_param_lr = np.linspace(1, len(self.param_lr_log), len(self.param_lr_log))
        x_arch_lr = np.linspace(1, len(self.arch_lr_log), len(self.arch_lr_log))
        x = (x_param_lr, x_arch_lr)

        # y data
        y_param_lr = self.param_lr_log[:]
        y_arch_lr = self.arch_lr_log[:]
        y = (y_param_lr, y_arch_lr)

        # axis limits
        x_limit = [1, len(self.param_lr_log)]

        if np.isnan(self.param_lr_log[:]).all() and np.isnan(self.arch_lr_log[:]).all():
            y_limit = [0, 1]
        else:
            y_max = np.nanmax([np.nanmax(self.param_lr_log[:]), np.nanmax(self.arch_lr_log[:])])
            y_limit = [0, y_max]

        # axis labels
        x_label = "Learning Rate"
        y_label = "Epochs"

        # legend
        legend = ('Parameter LR', 'Architecture LR')

        # generate plot dictionary
        plot_dict = self._generate_plot_dict(type, x, y, x_limit, y_limit, x_label, y_label, legend)
        self._performance_plots[self._lr_plot_name] = plot_dict

    def update_loss_plot(self):

        if hasattr(self, 'train_error_log') is not True:
            return

        # type
        type = Plot_Types.LINE

        # x data
        x_train = np.linspace(1, len(self.train_error_log), len(self.train_error_log))
        x_valid = np.linspace(1, len(self.valid_error_log), len(self.valid_error_log))
        x = (x_train, x_valid)

        # y data
        y_train = self.train_error_log[:]
        y_valid = self.valid_error_log[:]
        y = (y_train, y_valid)

        # axis limits
        x_limit = [1, len(self.train_error_log)]

        if np.isnan(self.train_error_log[:]).all() and np.isnan(self.valid_error_log[:]).all():
            y_limit = [0, 1]
        else:
            y_max = np.nanmax([np.nanmax(self.train_error_log[:]), np.nanmax(self.valid_error_log[:])])
            y_limit = [0, y_max]

        # axis labels
        add_str = ""
        if isinstance(self.criterion , nn.MSELoss):
            add_str = " (MSE)"
        elif isinstance(self.criterion , nn.CrossEntropyLoss):
            add_str = " (Cross-Entropy)"
        else:
            add_str = ""
        y_label = "Loss" + add_str
        x_label = "Epochs"

        # legend
        legend = ('Training Loss', 'Validation Loss')

        # generate plot dictionary
        plot_dict = self._generate_plot_dict(type, x, y, x_limit, y_limit, x_label, y_label, legend)
        self._performance_plots[self._loss_plot_name] = plot_dict

    def update_model_fit_plots(self, object_of_study):

        if hasattr(self, 'model') is not True:
            return

        # get all possible plots
        (IV_list_1, IV_list_2, DV_list) = self.get_model_fit_plot_list(object_of_study)

        # for each plot
        for IV1, IV2, DV in zip(IV_list_1, IV_list_2, DV_list):
            IVs = [IV1, IV2]

            # generate model prediction
            n_variables = len(object_of_study.independent_variables)
            resolution = int(np.round(aer_config.max_data_points_simulated**(1/float(n_variables))))

            counterbalanced_input = object_of_study.get_counterbalanced_input(resolution)
            #y_prediction = self.model(counterbalanced_input).detach().numpy()
            y_prediction = model_formatted(self.model, counterbalanced_input, object_of_study).detach().numpy()
            if IV2 is None:  # prepare line plot
                # x_prediction = object_of_study.get_IVs_from_input(counterbalanced_input, IV1).detach().numpy().flatten()
                # y_prediction = object_of_study.average_DV_for_IVs(DV, IVs, counterbalanced_input.detach().numpy(), y_prediction)
                x_prediction, y_prediction = object_of_study.average_DV_for_IVs(DV, IVs, counterbalanced_input.detach().numpy(), y_prediction)
            else:
                x_prediction, y_prediction = object_of_study.average_DV_for_IVs(DV, IVs, counterbalanced_input.detach().numpy(), y_prediction)

            # get data points
            (input, output)  = object_of_study.get_dataset()
            if IV2 is None:  # prepare line plot
                x_data = object_of_study.get_IVs_from_input(input, IV1).detach().numpy().flatten()
            else:
                x_data = (object_of_study.get_IVs_from_input(input, IV1).detach().numpy().flatten(), object_of_study.get_IVs_from_input(input, IV2).detach().numpy().flatten())
            y_data = object_of_study.get_DV_from_output(output, DV).detach().numpy().flatten()

            # get highlighted data points from last experiment
            last_experiment_id = object_of_study.get_last_experiment_id()
            (input_highlighted, output_highlighted) = object_of_study.get_dataset(experiment_id=last_experiment_id)
            if IV2 is None:  # prepare line plot
                x_data_highlighted = object_of_study.get_IVs_from_input(input_highlighted, IV1).detach().numpy().flatten()
            else:
                x_data_highlighted = (object_of_study.get_IVs_from_input(input_highlighted, IV1).detach().numpy().flatten(), object_of_study.get_IVs_from_input(input_highlighted, IV2).detach().numpy().flatten())
            y_data_highlighted = object_of_study.get_DV_from_output(output_highlighted, DV).detach().numpy().flatten()

            # determine y limits
            # y_limit = [np.amin([np.amin(y_data.numpy()), np.amin(y_prediction.detach().numpy()) ]),
            #            np.amax([np.amax(y_data.numpy()), np.amax(y_prediction.detach().numpy()) ])]
            y_limit = [np.amin(y_data), np.amax(y_data)]

            # determine y_label
            y_label = DV.get_variable_label()

            # determine legend:
            legend = ('All Data', 'Prediction', 'Novel Data')

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
                plot_name = DV.get_name() + "(" + IV1.get_name() + ", " + IV2.get_name() + ")"

                # determine x limits
                x_limit = (object_of_study.get_variable_limits(IV1),
                           object_of_study.get_variable_limits(IV2))

                # determine x_labels
                x_label = (IV1.get_variable_label(), IV2.get_variable_label())

            plot_dict = self._generate_plot_dict(type, x=x_data, y=y_data, x_limit=x_limit, y_limit=y_limit, x_label=x_label, y_label=y_label,
                                     legend=legend, image=None, x_model=x_prediction, y_model=y_prediction, x_highlighted=x_data_highlighted,
                                     y_highlighted=y_data_highlighted)
            self._performance_plots[plot_name] = plot_dict

    def get_model_fit_plot_list(self, object_of_study):
        (IV_list_1, IV_list_2, DV_list) = object_of_study.get_plot_list()
        return (IV_list_1, IV_list_2, DV_list)


    def update_arch_weights_plots(self):

        if hasattr(self, 'architecture_weights_log') is False:
            return
        # type
        type = Plot_Types.LINE

        # x axis label
        x_label = "Epoch"

        # x axis limits
        x_limit = [1, self.architecture_weights_log.shape[0]]

        for edge in range(self.num_arch_edges):

            # plot name
            plot_name = "Edge " + str(edge)

            # y axis label
            y_label = "Edge Weight (" + str(edge) + ")"

            # x data
            x = list()

            # y data
            y = list()
            legend = list()

            # add line (y-data and legend) for each primitive
            for op_idx, operation in enumerate(self.arch_ops_labels):

                x.append(np.linspace(1, self.architecture_weights_log.shape[0], self.architecture_weights_log.shape[0]))
                y.append(self.architecture_weights_log[:, edge, op_idx])
                legend.append(operation)

            # y axis limits
            if np.isnan(self.architecture_weights_log[:, edge, :]).all():
                y_limit = [0, 1]
            else:
                y_limit = [np.nanmin(np.nanmin(self.architecture_weights_log[:, edge, :])),
                           np.nanmax(np.nanmax(self.architecture_weights_log[:, edge, :]))]
            if y_limit[0] == y_limit[1]:
                y_limit = [0, 1]

            # generate plot dictionary
            plot_dict = self._generate_plot_dict(type, x, y, x_limit, y_limit, x_label, y_label, legend)
            self._supplementary_plots[plot_name] = plot_dict

        return self._supplementary_plots

    def update_pattern_plot(self):

        # type
        type = Plot_Types.IMAGE

        target = self.target_pattern
        prediction = self.prediction_pattern

        if len(target) == 0:
            self._performance_plots[self._pattern_plot_name] = None
            return

        im = np.concatenate((target, prediction), axis=1)

        # seperator
        x = np.ones(target.shape[0]) * (target.shape[1] - 0.5)
        y = np.linspace(1, target.shape[0], target.shape[0])

        # axis labels
        x_label = "Output"
        y_label = "Pattern"

        # generate plot dictionary
        plot_dict = self._generate_plot_dict(type, x, y, x_label=x_label, y_label=y_label, image=im)
        self._performance_plots[self._pattern_plot_name] = plot_dict

    def _meta_parameters_to_str(self):
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        label = 'decay_' + str(arch_weight_decay_df) + '_k_' + str(num_graph_nodes) + '_seed_' + str(seed)
        return label

    def _meta_parameter_names_to_str_list(self):
        names = ('decay', 'k', 'seed')
        return names

    def _meta_parameter_values_to_str_list(self):
        [arch_weight_decay_df, num_graph_nodes, seed] = self.get_meta_parameters()
        values = (str(arch_weight_decay_df), str(num_graph_nodes), str(seed))
        return values

    def _eval_meta_parameters_to_str(self):
        [arch_sample_id, param_sample_id] = self.get_eval_meta_parameters()
        label = 'arch_' + str(arch_sample_id) + '_param_' + str(param_sample_id)
        return label

    def get_model_filename(self, arch_weight_decay_df, num_graph_nodes, seed):
        filename = utils.create_output_file_name(file_prefix='model',
                                                         theorist=self.theorist_name,
                                                         log_version=self.model_search_id,
                                                         weight_decay=arch_weight_decay_df,
                                                         k=num_graph_nodes,
                                                         seed=seed)
        return filename

    def get_model_weights_filename(self, arch_weight_decay_df, num_graph_nodes, seed):
        filename = utils.create_output_file_name(file_prefix='model_weights',
                                                         theorist=self.theorist_name,
                                                         log_version=self.model_search_id,
                                                         weight_decay=arch_weight_decay_df,
                                                         k=num_graph_nodes,
                                                         seed=seed)
        return filename

    def get_architecture_filename(self, arch_weight_decay_df, num_graph_nodes, seed):
        filename = utils.create_output_file_name(file_prefix='architecture_weights',
                                                         theorist=self.theorist_name,
                                                         log_version=self.model_search_id,
                                                         weight_decay=arch_weight_decay_df,
                                                         k=num_graph_nodes,
                                                         seed=seed)
        return filename

    def evaluate_architectures(self, object_of_study, train_queue, valid_queue, model, arch_weight_decay_df, num_graph_nodes, seed):

      criterion = self.criterion
      criterion_loss_log = list()
      model_name_log = list()
      arch_name_log = list()
      num_graph_node_log = list()

      if self.DARTS_type == DARTS_Type.ORIGINAL:
          sample_amp = darts_cfg.sample_amp
      elif self.DARTS_type == DARTS_Type.FAIR:
          sample_amp = darts_cfg.sample_amp_fair_darts

      # generate general model file name
      model_filename_gen = self.get_model_weights_filename(arch_weight_decay_df, num_graph_nodes, seed)
      summary_filename_gen = self.get_model_filename(arch_weight_decay_df, num_graph_nodes, seed)
      arch_filename_gen = self.get_architecture_filename(arch_weight_decay_df, num_graph_nodes, seed)

      max_num_architectures = model.max_alphas_normal().shape[0] * model.max_alphas_normal().shape[1]
      n_architectures_sampled = np.min([max_num_architectures, darts_cfg.n_architectures_sampled])

      # subsample models and retrain
      sampled_weights = list()
      for arch_sample_id in range(n_architectures_sampled):

          logging.info('architecture evaluation for sampled model: %d / %d', arch_sample_id+1, n_architectures_sampled)

          # sample architecture weights
          found_weights = False
          if(arch_sample_id == 0):
              candidate_weights = model.max_alphas_normal()
              found_weights = True
          else:
              candidate_weights = model.sample_alphas_normal(sample_amp=sample_amp,
                                                             air_darts_weight_threshold=darts_cfg.fair_darts_weight_threshold)

          arch_search_attempt = 0
          while found_weights is False:
                weights_are_novel = True
                for logged_weights in sampled_weights:
                    if torch.eq(logged_weights, candidate_weights).all():
                        weights_are_novel = False
                if weights_are_novel:
                    novel_weights = candidate_weights
                    found_weights = True
                else:
                    candidate_weights = model.sample_alphas_normal(sample_amp=sample_amp,
                                                                   fair_darts_weight_threshold=darts_cfg.fair_darts_weight_threshold)
                    if arch_search_attempt > darts_cfg.max_arch_search_attempts:
                        found_weights = True
                arch_search_attempt += 1

          # store sampled architecture weights
          sampled_weights.append(candidate_weights)

          for param_sample_id in range(darts_cfg.n_initializations_sampled):

              logging.info('parameter evaluation for sampled model: %d / %d', param_sample_id + 1,
                           darts_cfg.n_initializations_sampled)

              # reinitialize weights if desired
              if darts_cfg.reinitialize_weights:
                  new_model = Network(object_of_study.__get_output_dim__(), criterion, steps=int(num_graph_nodes),
                                      n_input_states=object_of_study.__get_input_dim__(),
                                      classifier_weight_decay=darts_cfg.classifier_weight_decay,
                                      darts_type=self.DARTS_type)
                  if darts_cfg.eval_custom_initialization:
                    new_model.apply(init_weights)
              else:
                  new_model = copy.deepcopy(model)

              new_model.fix_architecture(True, candidate_weights)

              # optimizer is standard stochastic gradient decent with some momentum and weight decay
              optimizer = torch.optim.SGD(
                  new_model.parameters(),
                  darts_cfg.eval_learning_rate,
                  momentum=darts_cfg.eval_momentum,
                  weight_decay=darts_cfg.eval_weight_decay)

              # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
              scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, float(darts_cfg.eval_epochs), eta_min=darts_cfg.learning_rate_min)

              new_model.train()  # Sets the module in training mode

              loss_log = list()
              # train model
              for epoch in range(darts_cfg.eval_epochs):

                  # get new learning rate
                  lr = scheduler.get_last_lr()[0]

                  # get input and target
                  input_search, target_search = next(iter(train_queue))
                  input = Variable(input_search, requires_grad=False)  # .cuda()
                  target = Variable(target_search, requires_grad=False)  # .cuda(async=True)

                  input, target = format_input_target(input, target, criterion)

                  # zero out gradients
                  optimizer.zero_grad()
                  # compute loss for the model
                  logits = new_model(input)
                  loss = criterion(logits, target)
                  # update gradients for model
                  loss.backward()
                  # clips the gradient norm
                  nn.utils.clip_grad_norm_(new_model.parameters(), darts_cfg.grad_clip)
                  # moves optimizer one step (applies gradients to weights)
                  optimizer.step()
                  # applies weight decay to classifier weights
                  new_model.apply_weight_decay_to_classifier(lr)

                  # if in debug mode, print loss during architecture evaluation
                  logging.info('epoch %d', epoch)
                  logging.info('criterion loss %f', loss)
                  loss_log.append(loss.detach().numpy())
                  # moves the annealing scheduler forward to determine new learning rate
                  scheduler.step()

              # import matplotlib.pyplot as plt
              # plt.clf()
              # plt.plot(loss_log)
              # plt.show()

              # evaluate model
              criterion_loss = infer(valid_queue, new_model, criterion, silent = True)
              criterion_loss_log.append(criterion_loss.numpy())

              # get model name
              model_filename = model_filename_gen + '_sample' + str(arch_sample_id) + '_' + str(param_sample_id)
              arch_filename = arch_filename_gen + '_sample' + str(arch_sample_id) + '_' + str(param_sample_id)
              model_filepath = os.path.join(self.results_weights_path, model_filename + '.pt')
              arch_filepath = os.path.join(self.results_weights_path, arch_filename + '.pt')
              model_graph_filepath = os.path.join(self.results_path, model_filename)
              model_name_log.append(model_filename)
              arch_name_log.append(arch_filename)
              num_graph_node_log.append(num_graph_nodes)
              genotype = new_model.genotype()

              # save model
              utils.save(new_model, model_filepath)
              torch.save(new_model.alphas_normal, arch_filepath)
              print('Saving model weights: ' + model_filepath)
              (n_params_total, n_params_base, param_list) = new_model.countParameters()
              viz.plot(genotype.normal, model_graph_filepath, viewFile=False, input_labels=object_of_study.__get_input_labels__(), param_list=param_list, full_label=True,
                       out_dim=object_of_study.__get_output_dim__(), out_fnc=utils.get_output_str(object_of_study.__get_output_type__()))
              print('Saving model graph: ' + model_graph_filepath)
              print('Saving architecture weights: ' + arch_filepath)

      # get name for csv log file
      model_filename_csv = summary_filename_gen + '.csv'
      model_filepath = os.path.join(self.results_path, model_filename_csv)

      # save csv file
      rows = zip(model_name_log, arch_name_log, num_graph_node_log, criterion_loss_log)
      with open(model_filepath, "w") as f:
          writer = csv.writer(f)
          writer.writerow([darts_cfg.csv_model_file_name, darts_cfg.csv_arch_file_name, darts_cfg.csv_num_graph_node, darts_cfg.csv_loss])
          for row in rows:
              writer.writerow(row)

      return model_filepath


def model_formatted(model, input, object_of_study):
    m = utils.get_output_format(object_of_study.__get_output_type__())
    output = model(input)
    output_formatted = m(output)
    return output_formatted

def format_input_target(input, target, criterion):

    if isinstance(criterion, nn.CrossEntropyLoss):
        target = target.squeeze()

    return (input, target)

# trains model for one architecture epoch
def train(train_queue, valid_queue, model, architect, criterion, optimizer, arch_updates_per_epoch=1, param_updates_per_epoch = 1):
  objs = utils.AvgrageMeter() # metric that averages

  objs_log = torch.zeros(arch_updates_per_epoch)

  for arch_step in range(arch_updates_per_epoch):
    # for step, (input, target) in enumerate(train_queue): # for every pattern

    model.train() # Sets the module in training mode
    # logging.info("architecture step: %d", arch_step)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False) #.cuda()
    target_search = Variable(target_search, requires_grad=False) #.cuda(async=True)

    input_search, target_search = format_input_target(input_search, target_search, criterion)

    # FIRST STEP: UPDATE ARCHITECTURE (ALPHA)
    architect.step(input_search, target_search, optimizer, unrolled=darts_cfg.unrolled)

    # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
    optimizer = torch.optim.SGD(
        model.parameters(),
        darts_cfg.learning_rate,
        momentum=darts_cfg.momentum,
        weight_decay=darts_cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(param_updates_per_epoch), eta_min=darts_cfg.learning_rate_min)

    # SECOND STEP: UPDATE MODEL PARAMETERS (W)
    for param_step in range(param_updates_per_epoch):

      # get new learning rate
      lr = scheduler.get_last_lr()[0]
      # log new learning rate
      # logging.info('param_step: %d', param_step)
      # logging.info('learning rate: %e', lr)

      # get input and target
      input_search, target_search = next(iter(train_queue))
      input = Variable(input_search, requires_grad=False)  # .cuda()
      target = Variable(target_search, requires_grad=False)  # .cuda(async=True)

      input, target = format_input_target(input, target, criterion)

      # zero out gradients
      optimizer.zero_grad()
      # compute loss for the model
      logits = model(input)
      loss = criterion(logits, target)
      # update gradients for model
      loss.backward()
      # clips the gradient norm
      nn.utils.clip_grad_norm_(model.parameters(), darts_cfg.grad_clip)
      # moves optimizer one step (applies gradients to weights)
      optimizer.step()
      # applies weight decay to classifier weights
      model.apply_weight_decay_to_classifier(lr)

      # moves the annealing scheduler forward to determine new learning rate
      scheduler.step()

      # compute accuracy metrics
      n = input.size(0)
      objs.update(loss.data, n)

    objs_log[arch_step] = objs.avg

    if arch_step % darts_cfg.report_freq == 0:
      logging.info("architecture step (loss): %03d (%e)", arch_step, objs.avg)

  return objs.avg


# computes accuracy for validation set
def infer(valid_queue, model, criterion, silent = False):
  objs = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, requires_grad=True) #.cuda()
    target = Variable(target, requires_grad=True) #.cuda(async=True)

    input, target = format_input_target(input, target, criterion)

    logits = model(input)
    loss = criterion(logits, target)

    n = input.size(0)
    objs.update(loss.data, n)

    if silent is False:
        if step % darts_cfg.report_freq == 0:
          logging.info('architecture step (accuracy): %03d (%e)', step, objs.avg)

  return objs.avg

def init_weights(m):
    if type(m) == nn.Linear:
        # uniform initialization
        if darts_cfg.init_method == darts_cfg.InitMethod.UNIFORM:
            nn.init.uniform_(m.weight, a=darts_cfg.init_uniform_interval[0], b=darts_cfg.init_uniform_interval[1])
            if m.bias is not None:
                nn.init.uniform_(m.bias, a=darts_cfg.init_uniform_interval[0], b=darts_cfg.init_uniform_interval[1])

        # normal initialization
        elif darts_cfg.init_method == darts_cfg.InitMethod.NORMAL:
            nn.init.normal_(m.weight, mean=darts_cfg.init_normal_mean, std=darts_cfg.init_normal_std)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=darts_cfg.init_normal_mean, std=darts_cfg.init_normal_std)



# def architecture_search(self, object_of_study, arch_weight_decay_df, num_graph_nodes, seed):
#     # initializes the model given number of channels, output classes and the training criterion
#     model = Network(object_of_study.__get_output_dim__(), self.criterion, steps=int(num_graph_nodes),
#                     n_input_states=object_of_study.__get_input_dim__(),
#                     classifier_weight_decay=darts_cfg.classifier_weight_decay)
#
#     # log size of parameter space
#     logging.info("param size: %fMB", utils.count_parameters_in_MB(model))
#
#     # optimizer is standard stochastic gradient decent with some momentum and weight decay
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         darts_cfg.learning_rate,
#         momentum=darts_cfg.momentum,
#         weight_decay=darts_cfg.weight_decay)
#
#     # determine training set
#     train_data = object_of_study
#     num_train = len(train_data)  # number of patterns
#     indices = list(range(num_train))  # indices of all patterns
#     split = int(np.floor(darts_cfg.train_portion * num_train))  # size of training set
#
#     # combine the training set with a sampler, and provides an iterable over the training set
#     train_queue = torch.utils.data.DataLoader(
#         train_data, batch_size=darts_cfg.batch_size,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#         pin_memory=True, num_workers=0)
#
#     # combine the validation set with a sampler, and provides an iterable over the validation set
#     valid_queue = torch.utils.data.DataLoader(
#         train_data, batch_size=darts_cfg.batch_size,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
#         pin_memory=True, num_workers=0)
#
#     # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, float(self.model_search_epochs), eta_min=darts_cfg.learning_rate_min)
#
#     # generate an architecture of the model
#     architect = Architect(model, darts_cfg)
#
#     # plot variables
#     self.num_arch_edges = model.alphas_normal.data.shape[0]  # number of architecture edges
#     self.num_arch_ops = model.alphas_normal.data.shape[1]  # number of operations
#     self.arch_ops_labels = PRIMITIVES  # operations
#     self.train_error_log = np.empty((self.model_search_epochs, 1))  # log training error
#     self.valid_error_log = np.empty((self.model_search_epochs, 1))  # log validation error
#     self.train_error_log[:] = np.nan
#     self.valid_error_log[:] = np.nan
#     self.architecture_weights_log = np.empty(
#         (self.model_search_epochs, self.num_arch_edges, self.num_arch_ops))  # log architecture weights
#
#     graph_filename = utils.create_output_file_name(file_prefix=darts_cfg.graph_filename,
#                                                    log_version=self.model_search_id,
#                                                    weight_decay=arch_weight_decay_df,
#                                                    k=num_graph_nodes,
#                                                    seed=seed)
#     graph_filepath = os.path.join(self.results_path, graph_filename)
#
#     # architecture search loop
#     for epoch in range(self.model_search_epochs):
#         # get new learning rate
#         lr = scheduler.get_last_lr()[0]
#         # log new learning rate
#         logging.info('epoch: %d', epoch)
#         logging.info('learning rate: %e', lr)
#
#         # returns the genotype of the model
#         genotype = model.genotype()
#         # logs the genotype of the model
#         logging.info('genotype: %s', genotype)
#
#         # prints and log weights of the normal and reduced architecture
#         print(F.softmax(model.alphas_normal, dim=-1))
#
#         # training (for one epoch)
#         train_obj = train(train_queue, valid_queue, model, architect, self.criterion, optimizer, lr,
#                           darts_cfg.arch_updates_per_epoch, darts_cfg.param_updates_per_epoch)
#         # log accuracy on training set
#         logging.info('training accuracy: %f', train_obj)
#
#         # validation (for current epoch)
#         valid_obj = infer(valid_queue, model, self.criterion)
#         # log accuracy on validation set
#         logging.info('validation accuracy: %f', valid_obj)
#
#         # moves the annealing scheduler forward to determine new learning rate
#         scheduler.step()
#
#         # log data
#         self.architecture_weights_log[epoch, :, :] = torch.nn.functional.softmax(model.alphas_normal,
#                                                                                  dim=-1).data.numpy()
#         self.train_error_log[epoch] = train_obj
#         self.valid_error_log[epoch] = valid_obj
#
#     # save model plot
#     genotype = model.genotype()
#     viz.plot(genotype.normal, graph_filepath, fileFormat='png',
#              input_labels=object_of_study.__get_input_labels__())
#
#     # stores the model and architecture
#     model_filename = self.get_model_weights_filename(arch_weight_decay_df, num_graph_nodes, seed)
#     arch_filename = self.get_architecture_filename(arch_weight_decay_df, num_graph_nodes, seed)
#
#     model_filepath = os.path.join(self.results_path, model_filename + '.pt')
#     arch_filepath = os.path.join(self.results_path, arch_filename + '.pt')
#
#     utils.save(model, model_filepath)
#     torch.save(model.alphas_normal, arch_filepath)
#
#     model_eval_filepath = self.evaluate_architectures(object_of_study, train_queue, valid_queue, model,
#                                                       arch_weight_decay_df, num_graph_nodes, seed)
#
#     return model_eval_filepath