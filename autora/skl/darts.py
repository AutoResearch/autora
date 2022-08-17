import copy
import logging
from dataclasses import dataclass
from functools import partial
from itertools import cycle
from typing import Callable, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.utils
import torch.utils.data
import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from autora.theorist.darts.architect import Architect
from autora.theorist.darts.dataset import darts_dataset_from_ndarray
from autora.theorist.darts.model_search import DARTSType, Network
from autora.theorist.darts.operations import PRIMITIVES
from autora.theorist.darts.utils import (
    AvgrageMeter,
    format_input_target,
    get_loss_function,
    get_output_format,
    get_output_str,
)
from autora.theorist.darts.visualize import darts_model_plot
from autora.variable import ValueType

_logger = logging.getLogger(__name__)

progress_indicator = tqdm.auto.tqdm

SAMPLING_STRATEGIES = Literal["max", "sample"]
IMPLEMENTED_DARTS_TYPES = Literal["original", "fair"]
IMPLEMENTED_OUTPUT_TYPES = Literal[
    "real",
    "sigmoid",
    "probability",
    "probability_sample",
    "probability_distribution",
]


@dataclass(frozen=True)
class _DARTSResult:
    """A container for passing fitted DARTS results around."""

    network: Network
    model: torch.nn.Module
    model_sampler: Callable[[SAMPLING_STRATEGIES], torch.nn.Module]


def _general_darts(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 20,
    num_graph_nodes: int = 2,
    output_type: IMPLEMENTED_OUTPUT_TYPES = "real",
    classifier_weight_decay: float = 1e-2,
    darts_type: IMPLEMENTED_DARTS_TYPES = "original",
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
    primitives: Sequence[str] = PRIMITIVES,
) -> _DARTSResult:
    """
    Function to implement the DARTS optimization, given a fixed architecture and input data.
    """

    _logger.info("Starting fit initialization")

    data_loader, input_dimensions, output_dimensions = _get_data_loader(
        X=X,
        y=y,
        batch_size=batch_size,
    )

    criterion = get_loss_function(ValueType(output_type))
    output_function = get_output_format(ValueType(output_type))

    network = Network(
        num_classes=output_dimensions,
        criterion=criterion,
        steps=num_graph_nodes,
        n_input_states=input_dimensions,
        classifier_weight_decay=classifier_weight_decay,
        darts_type=DARTSType(darts_type),
        primitives=primitives,
    )
    if init_weights_function is not None:
        network.apply(init_weights_function)

    # Generate the architecture of the model
    architect = Architect(
        network,
        momentum=momentum,
        arch_weight_decay=arch_weight_decay,
        arch_weight_decay_df=arch_weight_decay_df,
        arch_weight_decay_base=arch_weight_decay_base,
        fair_darts_loss_weight=fair_darts_loss_weight,
        arch_learning_rate=arch_learning_rate,
    )

    optimizer = torch.optim.SGD(
        params=network.parameters(),
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

    _logger.info("Starting fit.")
    network.train()

    for epoch in progress_indicator(range(max_epochs)):

        _logger.debug(f"Running fit, epoch {epoch}")

        # Do the Architecture update

        # First reset the data iterator
        data_iterator = _get_data_iterator(data_loader)

        # Then run the arch optimization
        for arch_step in range(arch_updates_per_epoch):
            _logger.debug(
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

        # Then run the param optimization
        coefficient_optimizer(network)

    model = _generate_model(
        network=network,
        coefficient_optimizer=coefficient_optimizer,
        output_function=output_function,
        sampling_strategy="max",
    )
    model_sampler = partial(
        _generate_model,
        network=network,
        coefficient_optimizer=coefficient_optimizer,
        output_function=output_function,
    )

    results = _DARTSResult(model=model, model_sampler=model_sampler, network=network)

    return results


def _optimize_coefficients(
    network: Network,
    criterion: Callable,
    data_loader: torch.utils.data.DataLoader,
    grad_clip: bool,
    optimizer: torch.optim.Optimizer,
    param_updates_per_epoch: int,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
):
    """
    Function to optimize the coefficients of a DARTS Network.

    Warning: This modifies the coefficients of the Network in place.
    """

    data_iterator = _get_data_iterator(data_loader)

    objs = AvgrageMeter()

    for param_step in range(param_updates_per_epoch):
        _logger.debug(f"Running parameter update, " f"param: {param_step}")

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

    X_, y_ = check_X_y(X, y, ensure_2d=True, multi_output=True)

    if y_.ndim == 1:
        y_ = y_.reshape((y_.size, 1))

    input_dimensions = X_.shape[1]
    output_dimensions = y_.shape[1]

    experimental_data = darts_dataset_from_ndarray(X_, y_)

    data_loader = torch.utils.data.DataLoader(
        experimental_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    return data_loader, input_dimensions, output_dimensions


def _get_data_iterator(data_loader: torch.utils.data.DataLoader) -> Iterator:
    data_iterator = cycle(iter(data_loader))
    return data_iterator


def _get_next_input_target(data_iterator: Iterator, criterion: torch.nn.Module):
    input_search, target_search = next(data_iterator)

    input_var = torch.autograd.Variable(input_search, requires_grad=False)
    target_var = torch.autograd.Variable(target_search, requires_grad=False)

    input_fmt, target_fmt = format_input_target(
        input_var, target_var, criterion=criterion
    )
    return input_fmt, target_fmt


def _generate_model(
    sampling_strategy: SAMPLING_STRATEGIES,
    network: Network,
    coefficient_optimizer: Callable[[Network], None],
    output_function: torch.nn.Module,
):

    # Set edges in the network with the highest weights to 1, others to 0
    model_without_output_function = copy.deepcopy(network)

    if sampling_strategy == "max":
        new_weights = model_without_output_function.max_alphas_normal()
    elif sampling_strategy == "sample":
        new_weights = model_without_output_function.sample_alphas_normal()

    model_without_output_function.fix_architecture(True, new_weights=new_weights)

    # Re-optimize the parameters
    coefficient_optimizer(model_without_output_function)

    # Include the output function
    model = torch.nn.Sequential(model_without_output_function, output_function)

    return model


class DARTSRegressor(BaseEstimator, RegressorMixin):
    """
    Differentiable ARchiTecture Search Regressor.

    DARTS finds a composition of functions and coefficients to minimize a loss function suitable for
    the dependent variable.

    This class is intended to be compatible with the
    [Scikit-Learn Estimator API](https://scikit-learn.org/stable/developers/develop.html).

    Examples:

        >>> import numpy as np
        >>> num_samples = 1000
        >>> X = np.linspace(start=0, stop=1, num=num_samples).reshape(-1, 1)
        >>> y = 15. * np.ones(num_samples)
        >>> estimator = DARTSRegressor(num_graph_nodes=1)
        >>> estimator = estimator.fit(X, y)
        >>> estimator.predict([[0.5]])
        array([[15.051043]], dtype=float32)


    Attributes:
        network_: represents the optimized network for the architecture search, without the
            output function
        model_: represents the best-fit model including the output function
            after sampling of the network to pick a single computation graph.
            By default, this is the computation graph with the maximum weights,
            but can be set to a graph based on a sample on the edge weights
            by running the `resample_model(sample_strategy="sample")` method.
            It can be reset by running the `resample_model(sample_strategy="max")` method.
        model_sampler_: a callable which generates versions of the model, either based on the
            maximum architecture weights or based on a sample over the architecture weights.



    """

    def __init__(
        self,
        batch_size: int = 64,
        num_graph_nodes: int = 2,
        classifier_weight_decay: float = 1e-2,
        darts_type: IMPLEMENTED_DARTS_TYPES = "original",
        init_weights_function: Optional[Callable] = None,
        learning_rate: float = 2.5e-2,
        learning_rate_min: float = 0.01,
        momentum: float = 9e-1,
        optimizer_weight_decay: float = 3e-4,
        param_updates_per_epoch: int = 10,
        arch_updates_per_epoch: int = 1,
        arch_weight_decay: float = 1e-4,
        arch_weight_decay_df: float = 3e-4,
        arch_weight_decay_base: float = 0.0,
        arch_learning_rate: float = 3e-3,
        fair_darts_loss_weight: int = 1,
        max_epochs: int = 10,
        grad_clip: float = 5,
        output_type: IMPLEMENTED_OUTPUT_TYPES = "real",
        primitives: Sequence[str] = PRIMITIVES,
    ) -> None:
        """
        Arguments:
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
            primitives: list of primitive operations used in the DARTS network,
                e.g., 'add', 'subtract', 'none'. For details, see
                [`autora.theorist.darts.operations`][autora.theorist.darts.operations]
        """

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

        self.primitives = primitives

        self.output_type = output_type
        self.darts_type = darts_type

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.network_: Optional[Network] = None
        self.model_: Optional[Network] = None
        self.model_sampler_: Optional[Callable[[SAMPLING_STRATEGIES], Network]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array

        Returns:
            self (DARTSRegressor): the fitted estimator
        """

        if self.output_type == "class":
            raise NotImplementedError(
                "Classification not implemented for DARTSRegressor."
            )

        params = self.get_params()

        fit_results = _general_darts(X=X, y=y, **params)
        self.X_ = X
        self.y_ = y
        self.network_ = fit_results.network
        self.model_ = fit_results.model
        self.model_sampler_ = fit_results.model_sampler
        return self

    def resample_model(
        self, sampling_strategy: SAMPLING_STRATEGIES = "sample"
    ) -> Network:
        """
        Generates a new model based on a sample of the architecture weights
        (`sampling_strategy="sample"`) or the maximum architecture weights
        (`sampling_strategy="max"`)

        Args:
            sampling_strategy:

        Returns:

        """
        check_is_fitted(self, attributes=["model_sampler_"])
        assert self.model_sampler_ is not None
        self.model_ = self.model_sampler_(sampling_strategy)
        return self.model_

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

        # First run the checks using the scikit-learn API, listing the key parameters
        check_is_fitted(self, attributes=["model_"])

        # Since self.model_ is initialized as None, mypy throws an error if we
        # just call self.model_(X) in the predict method, as it could still be none.
        # MyPy doesn't understand that the sklearn check_is_fitted function
        # ensures the self.model_ parameter is initialized and otherwise throws an error,
        # so we check that explicitly here and pass the model which can't be None.
        assert self.model_ is not None

        y_ = self.model_(torch.as_tensor(X_).float())
        y = y_.detach().numpy()

        return y

    def visualize_model(
        self,
        input_labels: Optional[Sequence[str]] = None,
    ):
        assert self.model_ is not None
        fitted_sampled_network = self.model_[0]

        genotype = Network.genotype(fitted_sampled_network).normal
        (
            _,
            _,
            param_list,
        ) = fitted_sampled_network.countParameters()

        assert self.X_ is not None

        if input_labels is not None:
            input_labels_ = tuple(input_labels)
        elif hasattr(self.X_, "columns"):
            input_labels_ = tuple(self.X_.columns)
        else:
            input_dim = 1 if self.X_.ndim == 1 else self.X_.shape[1]
            input_labels_ = tuple(f"x{i}" for i in range(input_dim))

        assert self.y_ is not None
        out_dim = 1 if self.y_.ndim == 1 else self.y_.shape[1]

        out_func = get_output_str(ValueType(self.output_type))

        # call to plot function
        graph = darts_model_plot(
            genotype=genotype,
            input_labels=input_labels_,
            param_list=param_list,
            full_label=True,
            out_dim=out_dim,
            out_fnc=out_func,
        )

        return graph
