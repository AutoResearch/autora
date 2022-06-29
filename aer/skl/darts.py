import logging
from types import SimpleNamespace
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.utils
from sklearn.utils.validation import check_X_y

import aer.config
import aer.object_of_study
from aer.object_of_study import VariableCollection, new_object_of_study
from aer.theorist.darts.architect import Architect
from aer.theorist.darts.model_search import DARTS_Type, Network
from aer.theorist.darts.utils import AvgrageMeter, get_loss_function
from aer.theorist.theorist_darts import format_input_target

logger = logging.getLogger(__name__)


class DARTS:
    def __init__(
        self,
        variable_collection: VariableCollection,
        darts_type: DARTS_Type = DARTS_Type.ORIGINAL,
        max_epochs: int = 100,
        param_updates_per_epoch: int = 20,
        arch_weight_decay: float = 1e-4,
        arch_weight_decay_df: float = 3e-4,
        arch_weight_decay_base: float = 0.0,
        arch_updates_per_epoch: int = 20,
        num_graph_nodes: int = 2,
        classifier_weight_decay: float = 1e-2,
        optimizer_weight_decay: float = 3e-4,
        learning_rate: float = 2.5e-2,
        learning_rate_min: float = 0.01,
        momentum: float = 9e-1,
        init_weights_function: Callable = None,
        fair_darts_loss_weight: int = 1,
        arch_learning_rate: float = 3e-3,
        grad_clip: float = 5,
    ):
        self.variable_collection = variable_collection

        self.max_epochs = max_epochs
        self.darts_type = darts_type
        self.arch_weight_decay_df = arch_weight_decay_df
        self.arch_weight_decay = arch_weight_decay
        self.arch_updates_per_epoch = arch_updates_per_epoch
        self.classifier_weight_decay = classifier_weight_decay
        self.optimizer_weight_decay = optimizer_weight_decay
        self.arch_weight_decay_base = arch_weight_decay_base
        self.num_graph_nodes = num_graph_nodes
        self.learning_rate = learning_rate
        self.arch_learning_rate = arch_learning_rate
        self.momentum = momentum
        self.fair_darts_loss_weight = fair_darts_loss_weight

        self.init_weights_function = init_weights_function

        self.param_updates_per_epoch = param_updates_per_epoch
        self.learning_rate_min = learning_rate_min
        self.grad_clip = grad_clip

        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        # Do whatever happens in theorist.init_model_search
        logger.info("Starting fit initialization")
        X_reshaped, y_reshaped = check_X_y(X, y)

        object_of_study = new_object_of_study(self.variable_collection)
        data_dict = {}
        for i, iv in enumerate(self.variable_collection.independent_variables):
            data_dict[iv.name] = X_reshaped[:, i]

        assert self.variable_collection.output_dimensions == 1, (
            f"too many output dimensions "
            f"({self.variable_collection.output_dimensions}), "
            f"only one supported"
        )
        data_dict[self.variable_collection.dependent_variables[0].name] = y_reshaped

        data_dict[aer.config.experiment_label] = np.zeros(y_reshaped.shape, dtype=int)

        object_of_study.add_data(data_dict)
        train_queue = torch.utils.data.DataLoader(object_of_study)
        train_iterator = iter(train_queue)

        def get_next_input_target(train_iterator: Iterator):
            input_search, target_search = next(train_iterator)
            input_var = torch.autograd.Variable(input_search, requires_grad=False)
            target_var = torch.autograd.Variable(target_search, requires_grad=False)

            input_fmt, target_fmt = format_input_target(
                input_var, target_var, criterion=criterion
            )
            return input_fmt, target_fmt

        criterion = get_loss_function(self.variable_collection.output_type)

        model = Network(
            num_classes=self.variable_collection.output_dimensions,
            criterion=criterion,
            steps=self.num_graph_nodes,
            n_input_states=self.variable_collection.input_dimensions,
            classifier_weight_decay=self.classifier_weight_decay,
            darts_type=self.darts_type,
        )

        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.optimizer_weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.param_updates_per_epoch,
            eta_min=self.learning_rate_min,
        )

        # initialize model
        if self.init_weights_function is not None:
            model.apply(self.init_weights_function)

        # Generate the architecture of the model
        architect = Architect(
            model,
            SimpleNamespace(
                momentum=self.momentum,
                arch_weight_decay=self.arch_weight_decay,
                arch_weight_decay_df=self.arch_weight_decay_df,
                arch_weight_decay_base=self.arch_weight_decay_base,
                fair_darts_loss_weight=self.fair_darts_loss_weight,
                arch_learning_rate=self.arch_learning_rate,
            ),
        )

        objs = AvgrageMeter()

        logger.info(f"Starting fit.")
        model.train()

        for epoch in range(self.max_epochs):

            logger.info(f"Running fit, epoch {epoch}")
            for arch_step in range(self.arch_updates_per_epoch):

                logger.info(
                    f"Running architecture update, epoch:arch {epoch}:{arch_step}"
                )

                input, target = get_next_input_target(train_iterator)

                architect.step(
                    input_valid=input,
                    target_valid=target,
                    network_optimizer=optimizer,
                    unrolled=False,
                )

                for param_step in range(self.param_updates_per_epoch):

                    logger.info(
                        f"Running parameter update, "
                        f"epoch:arch:param "
                        f"{epoch}:{arch_step}:{param_step}"
                    )

                    lr = scheduler.get_last_lr()[0]
                    input, target = get_next_input_target(train_iterator)
                    optimizer.zero_grad()

                    # compute loss for the model
                    logits = model(input)
                    loss = criterion(logits, target)

                    # update gradients for model
                    loss.backward()

                    # clips the gradient norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    # moves optimizer one step (applies gradients to weights)
                    optimizer.step()
                    # applies weight decay to classifier weights
                    model.apply_weight_decay_to_classifier(lr)

                    # moves the annealing scheduler forward to determine new learning rate
                    scheduler.step()

                    # compute accuracy metrics
                    n = input.size(0)
                    objs.update(loss.data, n)

        return self

    def predict(self, X):
        pass
