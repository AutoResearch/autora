import copy
import logging
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.utils
import torch.utils.data
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y

import aer.config
from aer.object_of_study import new_object_of_study
from aer.theorist.darts.architect import Architect
from aer.theorist.darts.model_search import DARTS_Type, Network
from aer.theorist.darts.utils import AvgrageMeter, get_loss_function
from aer.theorist.theorist_darts import format_input_target
from aer.variable import ValueType, Variable, VariableCollection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DARTSResult:
    """A container for passing fitted DARTS results around."""

    network_: Network
    model_: Network


def _general_darts(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 20,
    num_graph_nodes: int = 2,
    output_type: ValueType = ValueType.REAL,
    classifier_weight_decay: float = 1e-2,
    darts_type: DARTS_Type = DARTS_Type.ORIGINAL,
    init_weights_function: Optional[Callable] = None,
    learning_rate: float = 2.5e-2,
    learning_rate_min: float = 0.01,
    momentum: float = 9e-1,
    optimizer_weight_decay: float = 3e-4,
    param_updates_per_epoch: int = 20,
    arch_updates_per_epoch: int = 20,
    arch_weight_decay: float = 1e-4,
    arch_weight_decay_df: float = 3e-4,
    arch_weight_decay_base: float = 0.0,
    arch_learning_rate: float = 3e-3,
    fair_darts_loss_weight: int = 1,
    max_epochs: int = 100,
    grad_clip: float = 5,
) -> _DARTSResult:
    """
    Function to implement the DARTS optimization, given a fixed architecture.
    """

    logger.info("Starting fit initialization")

    data_loader, input_dimensions, output_dimensions = _get_data_loader(
        X=X,
        y=y,
        batch_size=batch_size,
    )

    criterion = get_loss_function(output_type)

    network_ = Network(
        num_classes=output_dimensions,
        criterion=criterion,
        steps=num_graph_nodes,
        n_input_states=input_dimensions,
        classifier_weight_decay=classifier_weight_decay,
        darts_type=darts_type,
    )
    if init_weights_function is not None:
        network_.apply(init_weights_function)

    # Generate the architecture of the model
    architect = Architect(
        network_,
        SimpleNamespace(
            momentum=momentum,
            arch_weight_decay=arch_weight_decay,
            arch_weight_decay_df=arch_weight_decay_df,
            arch_weight_decay_base=arch_weight_decay_base,
            fair_darts_loss_weight=fair_darts_loss_weight,
            arch_learning_rate=arch_learning_rate,
        ),
    )

    optimizer = torch.optim.SGD(
        params=network_.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=optimizer_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=param_updates_per_epoch,
        eta_min=learning_rate_min,
    )

    coefficient_optimizer = partial(
        _optimize_coefficients,
        criterion=criterion,
        data_loader=data_loader,
        grad_clip=grad_clip,
        optimizer=optimizer,
        param_updates_per_epoch=param_updates_per_epoch,
        scheduler=scheduler,
    )

    logger.info("Starting fit.")
    network_.train()

    for epoch in range(max_epochs):

        logger.info(f"Running fit, epoch {epoch}")

        # Do the Architecture update

        # First reset the data iterator
        data_iterator = _get_data_iterator(data_loader)

        # Then run the arch optimization
        for arch_step in range(arch_updates_per_epoch):
            logger.info(
                f"Running architecture update, "
                f"epoch: {epoch}, architecture: {arch_step}"
            )

            X_batch, y_batch = _get_next_input_target(
                data_iterator, criterion=criterion
            )

            architect.step(
                input_valid=X_batch,
                target_valid=y_batch,
                network_optimizer=optimizer,
                unrolled=False,
            )

        # Do the param update

        # The run the param optimization
        coefficient_optimizer(network_)

    # Create the final model

    # Set edges in the network with the highest weights to 1, others to 0
    model_ = copy.deepcopy(network_)
    new_weights = model_.max_alphas_normal()
    model_.fix_architecture(True, new_weights)

    # Re-optimize the parameters
    coefficient_optimizer(model_)

    results = _DARTSResult(model_=model_, network_=network_)

    return results


def _optimize_coefficients(
    network: Network,
    criterion: Callable,
    data_loader: torch.utils.data.DataLoader,
    grad_clip,
    optimizer,
    param_updates_per_epoch,
    scheduler,
):

    data_iterator = _get_data_iterator(data_loader)

    objs = AvgrageMeter()

    for param_step in range(param_updates_per_epoch):
        logger.info(f"Running parameter update, " f"param: {param_step}")

        lr = scheduler.get_last_lr()[0]
        X_batch, y_batch = _get_next_input_target(data_iterator, criterion=criterion)
        optimizer.zero_grad()

        # compute loss for the model
        logits = network(X_batch)
        loss = criterion(logits, y_batch)

        # update gradients for model
        loss.backward()

        # clips the gradient norm
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)

        # moves optimizer one step (applies gradients to weights)
        optimizer.step()

        # applies weight decay to classifier weights
        network.apply_weight_decay_to_classifier(lr)

        # moves the annealing scheduler forward to determine new learning rate
        scheduler.step()

        # compute accuracy metrics
        n = X_batch.size(0)
        objs.update(loss.data, n)


def _get_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Construct a minimal torch.utils.data.DataLoader for the input data."""

    X_, y_ = check_X_y(X, y)

    data_dict = dict()

    y_variables = [Variable("y")]
    data_dict["y"] = y_

    X_variables = []
    for i in range(X.shape[1]):
        xi_name = f"x{i}"
        data_dict[xi_name] = X_[:, i]
        X_variables.append(Variable(xi_name))

    data_dict[aer.config.experiment_label] = np.zeros(y_.shape, dtype=int)

    variable_collection = VariableCollection(
        independent_variables=X_variables, dependent_variables=y_variables
    )
    input_dimensions = variable_collection.input_dimensions
    output_dimensions = variable_collection.output_dimensions

    object_of_study = new_object_of_study(variable_collection)
    object_of_study.add_data(data_dict)

    data_loader = torch.utils.data.DataLoader(
        object_of_study,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    return data_loader, input_dimensions, output_dimensions


def _get_data_iterator(data_loader: torch.utils.data.DataLoader) -> Iterator:
    data_iterator = iter(data_loader)
    return data_iterator


def _get_next_input_target(data_iterator: Iterator, criterion: torch.nn.Module):
    input_search, target_search = next(data_iterator)

    input_var = torch.autograd.Variable(input_search, requires_grad=False)
    target_var = torch.autograd.Variable(target_search, requires_grad=False)

    input_fmt, target_fmt = format_input_target(
        input_var, target_var, criterion=criterion
    )
    return input_fmt, target_fmt


class DARTS(BaseEstimator, RegressorMixin):
    """
    Differentiable ARchiTecture Search.

    DARTS finds a composition of functions and coefficients to minimize a loss function suitable for
    the dependent variable.

    This class is intended to be compatible with the
    [Scikit-Learn Estimator API](https://scikit-learn.org/stable/developers/develop.html).

    Bug: Scikit-Learn-Incompatibility: Requires specifying "VariableCollection" in the constructor.
        This is not consistent with scikit-learn.
        TODO: Remove requirement for VariableCollection.

    Examples:

        >>> import numpy as np
        >>> from aer.variable import Variable
        >>> num_samples = 1000
        >>> X = np.linspace(start=0, stop=1, num=num_samples).reshape(-1, 1)
        >>> y = 15. * np.ones(num_samples)
        >>> estimator = DARTS(num_graph_nodes=1)
        >>> estimator = estimator.fit(X, y)
        >>> estimator.predict([[15.]])
        array([[15.051043]], dtype=float32)

    Arguments:
            variable_collection: name and domain of the indepdendent variables `X` and
               dependent variable `y`
            batch_size: number of observations to be used per update
            num_graph_nodes: number of intermediate nodes in the DARTS graph.
            classifier_weight_decay:
            darts_type:
            init_weights_function:
            learning_rate:
            learning_rate_min:
            momentum:
            optimizer_weight_decay:
            param_updates_per_epoch:
            arch_updates_per_epoch:
            arch_weight_decay:
            arch_weight_decay_df:
            arch_weight_decay_base:
            arch_learning_rate:
            fair_darts_loss_weight:
            max_epochs:
            grad_clip:

    Attributes:
        network_: represents the optimized network for the architecture search
        model_: represents the best-fit model after simplification of the best fit function


    """

    def __init__(
        self,
        batch_size: int = 64,
        num_graph_nodes: int = 2,
        classifier_weight_decay: float = 1e-2,
        darts_type: DARTS_Type = DARTS_Type.ORIGINAL,
        init_weights_function: Optional[Callable] = None,
        learning_rate: float = 2.5e-2,
        learning_rate_min: float = 0.01,
        momentum: float = 9e-1,
        optimizer_weight_decay: float = 3e-4,
        param_updates_per_epoch: int = 10,
        arch_updates_per_epoch: int = 10,
        arch_weight_decay: float = 1e-4,
        arch_weight_decay_df: float = 3e-4,
        arch_weight_decay_base: float = 0.0,
        arch_learning_rate: float = 3e-3,
        fair_darts_loss_weight: int = 1,
        max_epochs: int = 10,
        grad_clip: float = 5,
    ) -> None:

        self.batch_size = batch_size

        self.num_graph_nodes = num_graph_nodes
        self.classifier_weight_decay = classifier_weight_decay
        self.darts_type = darts_type
        self.init_weights_function = init_weights_function

        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.momentum = momentum
        self.optimizer_weight_decay = optimizer_weight_decay

        self.param_updates_per_epoch = param_updates_per_epoch

        self.arch_updates_per_epoch = arch_updates_per_epoch
        self.arch_weight_decay = arch_weight_decay
        self.arch_weight_decay_df = arch_weight_decay_df
        self.arch_weight_decay_base = arch_weight_decay_base
        self.arch_learning_rate = arch_learning_rate
        self.fair_darts_loss_weight = fair_darts_loss_weight

        self.max_epochs = max_epochs
        self.grad_clip = grad_clip

        self.network_: Network = Network(0, 0)
        self.model_: Network = Network(0, 0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array

        Returns:
            self (DARTS): the fitted estimator
        """
        params = self.get_params()
        fit_results = _general_darts(X=X, y=y, **params)
        self.network_ = fit_results.network_
        self.model_ = fit_results.model_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted model to a set of independent variables `X`,
        to give predictions for the dependent variable `y`.

        Arguments:
            X: independent variables in an n-dimensional array

        Returns:
            y: predicted dependent variable values
        """
        X_ = check_array(X)
        y_ = self.model_(torch.as_tensor(X_).float())
        y = y_.detach().numpy()
        return y
