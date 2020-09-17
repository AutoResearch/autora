from abc import ABC, abstractmethod
from AER_theorist.theorist import Theorist
from AER_theorist.object_of_study import Object_Of_Study
from AER_theorist.darts.model_search import Network

import torch
import logging
import numpy as np
import AER_theorist.darts.darts_config as darts_cfg
import AER_config as AER_cfg
import AER_theorist.darts.utils as utils


class Theorist_DARTS(Theorist, ABC):

    simulation_files = 'AER_theorist/darts/*.py'

    criterion = None

    def __init__(self, object_of_study: Object_Of_Study):
        super(Theorist_DARTS, self).__init__(object_of_study)

        # define loss function
        self.criterion = utils.get_loss_function(object_of_study.__get_output_type__())



    def initialize_model_search(self):
        super(Theorist_DARTS, self).initialize_model_search()

        # set configuration
        self._cfg = darts_cfg

        # log: gpu device, parameter configuration
        logging.info('gpu device = %d' % darts_cfg.gpu)
        logging.info("configuration = %s", darts_cfg)

    def commission_model_search(self):
        pass

    def run_model_search(self):
        self._log_version += 1

        # sets seeds
        np.random.seed(int(darts_cfg.seed))
        torch.manual_seed(int(darts_cfg.seed))

        # log: gpu device, parameter configuration
        logging.info('gpu device = %d' % darts_cfg.gpu)
        logging.info("configuration = %s", darts_cfg)

        # todo: loop over meta-parameters, then call architecture_search

        pass

    def architecture_search(self):
        # initializes the model given number of channels, output classes and the training criterion
        model = Network(args.outputDim, self.criterion, steps=int(args.num_graph_nodes),
                        n_input_states=int(args.inputDim),
                        classifier_weight_decay=args.classifier_weight_decay)
        pass