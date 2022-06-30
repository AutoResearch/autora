import logging
from types import SimpleNamespace
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn
import torch.nn.utils
import torch.utils.data
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y

import aer.config
import aer.object_of_study
from aer.object_of_study import VariableCollection, new_object_of_study
from aer.theorist.darts.architect import Architect
from aer.theorist.darts.model_search import DARTS_Type, Network
from aer.theorist.darts.utils import AvgrageMeter, get_loss_function
from aer.theorist.theorist_darts import format_input_target

logger = logging.getLogger(__name__)


class DARTS(BaseEstimator, RegressorMixin):
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
        batch_size: int = 20,
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
        self.batch_size = batch_size

        self.model_: Network = Network(0, 0)

    def fit(self, X: np.ndarray, y: np.ndarray):

        # Do whatever happens in theorist.init_model_search
        logger.info("Starting fit initialization")

        data_loader = get_data_loader(
            X=X,
            y=y,
            variable_collection=self.variable_collection,
            batch_size=self.batch_size,
        )

        criterion = get_loss_function(self.variable_collection.output_type)

        self.model_ = Network(
            num_classes=self.variable_collection.output_dimensions,
            criterion=criterion,
            steps=self.num_graph_nodes,
            n_input_states=self.variable_collection.input_dimensions,
            classifier_weight_decay=self.classifier_weight_decay,
            darts_type=self.darts_type,
        )

        optimizer = torch.optim.SGD(
            params=self.model_.parameters(),
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
            self.model_.apply(self.init_weights_function)

        # Generate the architecture of the model
        architect = Architect(
            self.model_,
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

        logger.info("Starting fit.")
        self.model_.train()

        for epoch in range(self.max_epochs):

            logger.info(f"Running fit, epoch {epoch}")

            # Do the Architecture update

            # First reset the data iterator
            data_iterator = get_data_iterator(data_loader)

            # Then run the arch optimization
            for arch_step in range(self.arch_updates_per_epoch):

                logger.info(
                    f"Running architecture update, "
                    f"epoch: {epoch}, architecture: {arch_step}"
                )

                X_batch, y_batch = get_next_input_target(
                    data_iterator, criterion=criterion
                )

                architect.step(
                    input_valid=X_batch,
                    target_valid=y_batch,
                    network_optimizer=optimizer,
                    unrolled=False,
                )

            # Do the param update
            # First reset the data iterator
            data_iterator = get_data_iterator(data_loader)

            # The run the param optimization
            for param_step in range(self.param_updates_per_epoch):

                logger.info(
                    f"Running parameter update, "
                    f"epoch:param "
                    f"epoch: {epoch}, param: {param_step}"
                )

                lr = scheduler.get_last_lr()[0]
                X_batch, y_batch = get_next_input_target(
                    data_iterator, criterion=criterion
                )
                optimizer.zero_grad()

                # compute loss for the model
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)

                # update gradients for model
                loss.backward()

                # clips the gradient norm
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)

                # moves optimizer one step (applies gradients to weights)
                optimizer.step()

                # applies weight decay to classifier weights
                self.model_.apply_weight_decay_to_classifier(lr)

                # moves the annealing scheduler forward to determine new learning rate
                scheduler.step()

                # compute accuracy metrics
                n = X_batch.size(0)
                objs.update(loss.data, n)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ = check_array(X)
        results_ = self.model_(torch.as_tensor(X_).float())
        results = results_.detach().numpy()
        return results


def get_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    variable_collection: VariableCollection,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    # Run checks and datatype conversions
    assert variable_collection.output_dimensions == 1, (
        f"too many output dimensions "
        f"({variable_collection.output_dimensions}), "
        f"only one supported"
    )

    X_, y_ = check_X_y(X, y)

    assert X.shape[1] == variable_collection.input_dimensions

    data_dict = dict()
    for i, iv in enumerate(variable_collection.independent_variables):
        data_dict[iv.name] = X_[:, i]
    data_dict[variable_collection.dependent_variables[0].name] = y_

    data_dict[aer.config.experiment_label] = np.zeros(y_.shape, dtype=int)

    object_of_study = new_object_of_study(variable_collection)
    object_of_study.add_data(data_dict)

    data_loader = torch.utils.data.DataLoader(
        object_of_study,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    return data_loader


def get_data_iterator(data_loader: torch.utils.data.DataLoader) -> Iterator:
    data_iterator = iter(data_loader)
    return data_iterator


def get_next_input_target(data_iterator: Iterator, criterion: torch.nn.Module):
    input_search, target_search = next(data_iterator)

    input_var = torch.autograd.Variable(input_search, requires_grad=False)
    target_var = torch.autograd.Variable(target_search, requires_grad=False)

    input_fmt, target_fmt = format_input_target(
        input_var, target_var, criterion=criterion
    )
    return input_fmt, target_fmt
