import copy
import logging
from dataclasses import dataclass
from itertools import cycle
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn
import torch.nn.utils
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from autora.theorist.darts import (
    PRIMITIVES,
    Architect,
    AvgrageMeter,
    DARTSType,
    Network,
    darts_dataset_from_ndarray,
    darts_model_plot,
    format_input_target,
    get_loss_function,
    get_output_format,
    get_output_str,
)
from autora.variable import ValueType

_logger = logging.getLogger(__name__)

_progress_indicator = tqdm.auto.tqdm

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


def _general_darts(
    X: np.ndarray,
    y: np.ndarray,
    network: Optional[Network] = None,
    batch_size: int = 20,
    num_graph_nodes: int = 2,
    output_type: IMPLEMENTED_OUTPUT_TYPES = "real",
    classifier_weight_decay: float = 1e-2,
    darts_type: IMPLEMENTED_DARTS_TYPES = "original",
    init_weights_function: Optional[Callable] = None,
    param_updates_per_epoch: int = 20,
    param_updates_for_sampled_model: int = 100,
    param_learning_rate_max: float = 2.5e-2,
    param_learning_rate_min: float = 0.01,
    param_momentum: float = 9e-1,
    param_weight_decay: float = 3e-4,
    arch_learning_rate_max: float = 3e-3,
    arch_updates_per_epoch: int = 20,
    arch_weight_decay: float = 1e-4,
    arch_weight_decay_df: float = 3e-4,
    arch_weight_decay_base: float = 0.0,
    arch_momentum: float = 9e-1,
    fair_darts_loss_weight: int = 1,
    max_epochs: int = 100,
    grad_clip: float = 5,
    primitives: Sequence[str] = PRIMITIVES,
    train_classifier_coefficients: bool = False,
    train_classifier_bias: bool = False,
    execution_monitor: Callable = (lambda *args, **kwargs: None),
    sampling_strategy: SAMPLING_STRATEGIES = "max",
) -> _DARTSResult:
    """
    Function to implement the DARTS optimization, given a fixed architecture and input data.

    Arguments:
        X: Input data.
        y: Target data.
        batch_size: Batch size for the data loader.
        num_graph_nodes: Number of nodes in the desired computation graph.
        output_type: Type of output function to use. This function is applied to transform
        the output of the mixture architecture.
        classifier_weight_decay: Weight decay for the classifier.
        darts_type: Type of DARTS to use ('original' or 'fair').
        init_weights_function: Function to initialize the parameters of each operation.
        param_learning_rate_max: Initial (maximum) learning rate for the operation parameters.
        param_learning_rate_min: Final (minimum) learning rate for the operation parameters.
        param_momentum: Momentum for the operation parameters.
        param_weight_decay: Weight decay for the operation parameters.
        param_updates_per_epoch: Number of updates to perform per epoch.
        for the operation parameters.
        arch_learning_rate_max: Initial (maximum) learning rate for the architecture.
        arch_updates_per_epoch: Number of architecture weight updates to perform per epoch.
        arch_weight_decay: Weight decay for the architecture weights.
        arch_weight_decay_df: An additional weight decay that scales with the number of parameters
        (degrees of freedom) in the operation. The higher this weight decay, the more DARTS will
        prefer simple operations.
        arch_weight_decay_base: A base weight decay that is added to the scaled weight decay.
        arch_momentum: Momentum for the architecture weights.
        fair_darts_loss_weight: Weight of the loss in fair darts which forces architecture weights
        to become either 0 or 1.
        max_epochs: Maximum number of epochs to train for.
        grad_clip: Gradient clipping value for updating the parameters of the operations.
        primitives: List of primitives (operations) to use.
        train_classifier_coefficients: Whether to train the coefficients of the classifier.
        train_classifier_bias: Whether to train the bias of the classifier.
        execution_monitor: Function to monitor the execution of the model.

    Returns:
        A _DARTSResult object containing the fitted model and the network architecture.
    """

    _logger.info("Starting fit initialization")

    data_loader, input_dimensions, output_dimensions = _get_data_loader(
        X=X,
        y=y,
        batch_size=batch_size,
    )

    criterion = get_loss_function(ValueType(output_type))
    output_function = get_output_format(ValueType(output_type))

    if network is None:
        network = Network(
            num_classes=output_dimensions,
            criterion=criterion,
            steps=num_graph_nodes,
            n_input_states=input_dimensions,
            classifier_weight_decay=classifier_weight_decay,
            darts_type=DARTSType(darts_type),
            primitives=primitives,
            train_classifier_coefficients=train_classifier_coefficients,
            train_classifier_bias=train_classifier_bias,
        )

    if init_weights_function is not None:
        network.apply(init_weights_function)

    # Generate the architecture of the model
    architect = Architect(
        network,
        arch_momentum=arch_momentum,
        arch_weight_decay=arch_weight_decay,
        arch_weight_decay_df=arch_weight_decay_df,
        arch_weight_decay_base=arch_weight_decay_base,
        fair_darts_loss_weight=fair_darts_loss_weight,
        arch_learning_rate_max=arch_learning_rate_max,
    )

    _logger.info("Starting fit.")
    network.train()

    for epoch in _progress_indicator(range(max_epochs)):

        _logger.debug(f"Running fit, epoch {epoch}")

        data_iterator = _get_data_iterator(data_loader)

        # Do the Architecture update
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
                network_optimizer=architect.optimizer,
                unrolled=False,
            )

        # Then run the param optimization
        _optimize_coefficients(
            network=network,
            criterion=criterion,
            data_loader=data_loader,
            grad_clip=grad_clip,
            param_learning_rate_max=param_learning_rate_max,
            param_learning_rate_min=param_learning_rate_min,
            param_momentum=param_momentum,
            param_update_steps=param_updates_per_epoch,
            param_weight_decay=param_weight_decay,
        )

        execution_monitor(**locals())

    model = _generate_model(
        network_=network,
        output_type=output_type,
        sampling_strategy=sampling_strategy,
        data_loader=data_loader,
        param_update_steps=param_updates_for_sampled_model,
        param_learning_rate_max=param_learning_rate_max,
        param_learning_rate_min=param_learning_rate_min,
        param_momentum=param_momentum,
        param_weight_decay=param_weight_decay,
        grad_clip=grad_clip,
    )

    results = _DARTSResult(model=model, network=network)

    return results


def _optimize_coefficients(
    network: Network,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    grad_clip: float,
    param_learning_rate_max: float,
    param_learning_rate_min: float,
    param_momentum: float,
    param_update_steps: int,
    param_weight_decay: float,
):
    """
    Function to optimize the coefficients of a DARTS Network.

    Warning: This modifies the coefficients of the Network in place.

    Arguments:
        network: The DARTS Network to optimize the coefficients of.
        criterion: The loss function to use.
        data_loader: The data loader to use for the optimization.
        grad_clip: Whether to clip the gradients.
        param_update_steps: The number of parameter update steps to perform.
        param_learning_rate_max: Initial (maximum) learning rate for the operation parameters.
        param_learning_rate_min: Final (minimum) learning rate for the operation parameters.
        param_momentum: Momentum for the operation parameters.
        param_weight_decay: Weight decay for the operation parameters.
    """
    optimizer = torch.optim.SGD(
        params=network.parameters(),
        lr=param_learning_rate_max,
        momentum=param_momentum,
        weight_decay=param_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=param_update_steps,
        eta_min=param_learning_rate_min,
    )

    data_iterator = _get_data_iterator(data_loader)

    objs = AvgrageMeter()

    if network.count_parameters()[0] == 0:
        return

    for param_step in range(param_update_steps):
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
    """Construct a minimal torch.utils.data.DataLoader for the input data.

    Arguments:
        X: The input data.
        y: The target data.
        batch_size: The batch size to use.

    Returns:
        A torch.utils.data.DataLoader for the input data.
    """

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
    """Get an iterator for the data loader.

    Arguments:
        data_loader: The data loader to get the iterator for.

    Returns:
        An iterator for the data loader.
    """
    data_iterator = cycle(iter(data_loader))
    return data_iterator


def _get_next_input_target(
    data_iterator: Iterator, criterion: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the next input and target from the data iterator.
    Args:
        data_iterator: The data iterator to get the next input and target from.
        criterion: The loss function to use.

    Returns:
        The next input and target from the data iterator.

    """
    input_search, target_search = next(data_iterator)

    input_var = torch.autograd.Variable(input_search, requires_grad=False)
    target_var = torch.autograd.Variable(target_search, requires_grad=False)

    input_fmt, target_fmt = format_input_target(
        input_var, target_var, criterion=criterion
    )
    return input_fmt, target_fmt


def _generate_model(
    network_: Network,
    output_type: IMPLEMENTED_OUTPUT_TYPES,
    sampling_strategy: SAMPLING_STRATEGIES,
    data_loader: torch.utils.data.DataLoader,
    param_update_steps: int,
    param_learning_rate_max: float,
    param_learning_rate_min: float,
    param_momentum: float,
    param_weight_decay: float,
    grad_clip: float,
) -> Network:
    """
    Generate a model architecture from mixed DARTS model.

    Arguments:
        sampling_strategy: The sampling strategy used to pick the operations
        based on the trained architecture weights (e.g. "max", "sample").
        network: The mixed DARTS model.
        coefficient_optimizer: The function to optimize the coefficients of the trained model
        output_type: The output value type that is used for the output of the sampled model.
        param_update_steps: The number of parameter update steps to perform.
        param_learning_rate_max: Initial (maximum) learning rate for the operation parameters.
        param_learning_rate_min: Final (minimum) learning rate for the operation parameters.
        param_momentum: Momentum for the operation parameters.
        param_weight_decay: Weight decay for the operation parameters.

    Returns:
        A model architecture that is a combination of the trained model and the output function.
    """
    criterion = get_loss_function(ValueType(output_type))
    output_function = get_output_format(ValueType(output_type))

    # Set edges in the network with the highest weights to 1, others to 0
    model_without_output_function = copy.deepcopy(network_)

    if sampling_strategy == "max":
        new_weights = model_without_output_function.max_alphas_normal()
    elif sampling_strategy == "sample":
        new_weights = model_without_output_function.sample_alphas_normal()

    model_without_output_function.fix_architecture(True, new_weights=new_weights)

    # Re-optimize the parameters

    _optimize_coefficients(
        model_without_output_function,
        criterion=criterion,
        data_loader=data_loader,
        grad_clip=grad_clip,
        param_learning_rate_max=param_learning_rate_max,
        param_learning_rate_min=param_learning_rate_min,
        param_momentum=param_momentum,
        param_update_steps=param_update_steps,
        param_weight_decay=param_weight_decay,
    )

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



    """

    def __init__(
        self,
        batch_size: int = 64,
        num_graph_nodes: int = 2,
        output_type: IMPLEMENTED_OUTPUT_TYPES = "real",
        classifier_weight_decay: float = 1e-2,
        darts_type: IMPLEMENTED_DARTS_TYPES = "original",
        init_weights_function: Optional[Callable] = None,
        param_updates_per_epoch: int = 10,
        param_updates_for_sampled_model: int = 100,
        param_learning_rate_max: float = 2.5e-2,
        param_learning_rate_min: float = 0.01,
        param_momentum: float = 9e-1,
        param_weight_decay: float = 3e-4,
        arch_updates_per_epoch: int = 1,
        arch_learning_rate_max: float = 3e-3,
        arch_weight_decay: float = 1e-4,
        arch_weight_decay_df: float = 3e-4,
        arch_weight_decay_base: float = 0.0,
        arch_momentum: float = 9e-1,
        fair_darts_loss_weight: int = 1,
        max_epochs: int = 10,
        grad_clip: float = 5,
        primitives: Sequence[str] = PRIMITIVES,
        train_classifier_coefficients: bool = False,
        train_classifier_bias: bool = False,
        execution_monitor: Callable = (lambda *args, **kwargs: None),
        sampling_strategy: SAMPLING_STRATEGIES = "max",
    ) -> None:
        """
        Initializes the DARTSRegressor.

        Arguments:
            batch_size: Batch size for the data loader.
            num_graph_nodes: Number of nodes in the desired computation graph.
            output_type: Type of output function to use. This function is applied to transform
                the output of the mixture architecture.
            classifier_weight_decay: Weight decay for the classifier.
            darts_type: Type of DARTS to use ('original' or 'fair').
            init_weights_function: Function to initialize the parameters of each operation.
            param_updates_per_epoch: Number of updates to perform per epoch.
                for the operation parameters.
            param_learning_rate_max: Initial (maximum) learning rate for the operation parameters.
            param_learning_rate_min: Final (minimum) learning rate for the operation parameters.
            param_momentum: Momentum for the operation parameters.
            param_weight_decay: Weight decay for the operation parameters.
            arch_updates_per_epoch: Number of architecture weight updates to perform per epoch.
            arch_learning_rate_max: Initial (maximum) learning rate for the architecture.
            arch_weight_decay: Weight decay for the architecture weights.
            arch_weight_decay_df: An additional weight decay that scales with the number of
                parameters (degrees of freedom) in the operation. The higher this weight decay,
                the more DARTS will prefer simple operations.
            arch_weight_decay_base: A base weight decay that is added to the scaled weight decay.
                arch_momentum: Momentum for the architecture weights.
            fair_darts_loss_weight: Weight of the loss in fair darts which forces architecture
                weights to become either 0 or 1.
            max_epochs: Maximum number of epochs to train for.
            grad_clip: Gradient clipping value for updating the parameters of the operations.
            primitives: List of primitives (operations) to use.
            train_classifier_coefficients: Whether to train the coefficients of the classifier.
            train_classifier_bias: Whether to train the bias of the classifier.
            execution_monitor: Function to monitor the execution of the model.
            primitives: list of primitive operations used in the DARTS network,
                e.g., 'add', 'subtract', 'none'. For details, see
                [`autora.theorist.darts.operations`][autora.theorist.darts.operations]
        """

        self.batch_size = batch_size

        self.num_graph_nodes = num_graph_nodes
        self.classifier_weight_decay = classifier_weight_decay
        self.darts_type = darts_type
        self.init_weights_function = init_weights_function

        self.param_updates_per_epoch = param_updates_per_epoch
        self.param_updates_for_sampled_model = param_updates_for_sampled_model

        self.param_learning_rate_max = param_learning_rate_max
        self.param_learning_rate_min = param_learning_rate_min
        self.param_momentum = param_momentum
        self.arch_momentum = arch_momentum
        self.param_weight_decay = param_weight_decay

        self.arch_updates_per_epoch = arch_updates_per_epoch
        self.arch_weight_decay = arch_weight_decay
        self.arch_weight_decay_df = arch_weight_decay_df
        self.arch_weight_decay_base = arch_weight_decay_base
        self.arch_learning_rate_max = arch_learning_rate_max
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

        self.train_classifier_coefficients = train_classifier_coefficients
        self.train_classifier_bias = train_classifier_bias

        self.execution_monitor = execution_monitor

        self.sampling_strategy = sampling_strategy

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

        fit_results = _general_darts(X=X, y=y, network=self.network_, **params)
        self.X_ = X
        self.y_ = y
        self.network_ = fit_results.network
        self.model_ = fit_results.model
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
        """
        Visualizes the model architecture as a graph.

        Arguments:
            input_labels: labels for the input nodes

        """

        check_is_fitted(self, attributes=["model_"])
        assert self.model_ is not None
        fitted_sampled_network = self.model_[0]

        genotype = Network.genotype(fitted_sampled_network).normal
        (
            _,
            _,
            param_list,
        ) = fitted_sampled_network.count_parameters()

        if input_labels is not None:
            input_labels_ = tuple(input_labels)
        else:
            input_labels_ = self._get_input_labels()

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

    def _get_input_labels(self):
        """
        Returns the input labels for the model.

        Returns:
            input_labels: labels for the input nodes

        """
        return self._get_labels(self.X_, "x")

    def _get_output_labels(self):
        """
        Returns the output labels for the model.

        Returns:
            output_labels: labels for the output nodes

        """
        return self._get_labels(self.y_, "y")

    def _get_labels(
        self, data: Optional[np.ndarray], default_label: str
    ) -> Sequence[str]:
        """
        Returns the labels for the model.

        Arguments:
            data: data to get labels for
            default_label: default label to use if no labels are provided

        Returns:
            labels: labels for the model

        """
        assert data is not None

        if hasattr(data, "columns"):  # it's a dataframe with column names
            labels_ = tuple(data.columns)
        elif (
            hasattr(data, "name") and len(data.shape) == 1
        ):  # it's a single series with a single name
            labels_ = (data.name,)

        else:
            dim = 1 if data.ndim == 1 else data.shape[1]
            labels_ = tuple(f"{default_label}{i+1}" for i in range(dim))
        return labels_

    def model_repr(
        self,
        input_labels: Optional[Sequence[str]] = None,
        output_labels: Optional[Sequence[str]] = None,
        output_function_label: str = "",
        decimals_to_display: int = 2,
        output_format: Literal["latex", "console"] = "console",
    ) -> str:
        """
        Prints the equations of the model architecture.

        Args:
            input_labels: which names to use for the independent variables (X)
            output_labels: which names to use for the dependent variables (y)
            output_function_label: name to use for the output transformation
            decimals_to_display: amount of rounding for the coefficient values
            output_format: whether the output should be formatted for
                the command line (`console`) or as equations in a latex file (`latex`)

        Returns:
            The equations of the model architecture

        """
        assert self.model_ is not None
        fitted_sampled_network: Network = self.model_[0]

        if input_labels is None:
            input_labels_ = self._get_input_labels()
        else:
            input_labels_ = input_labels

        if output_labels is None:
            output_labels_ = self._get_output_labels()
        else:
            output_labels_ = output_labels

        edge_list = fitted_sampled_network.architecture_to_str_list(
            input_labels=input_labels_,
            output_labels=output_labels_,
            output_function_label=output_function_label,
            decimals_to_display=decimals_to_display,
            output_format=output_format,
        )

        model_repr_ = "\n".join(["Model:"] + edge_list)
        return model_repr_


class DARTSExecutionMonitor:
    """
    A monitor of the execution of the DARTS algorithm.
    """

    def __init__(self):
        """
        Initializes the execution monitor.
        """
        self.arch_weight_history = list()
        self.loss_history = list()
        self.epoch_history = list()
        self.primitives = list()

    def execution_monitor(
        self,
        network: Network,
        architect: Architect,
        epoch: int,
        **kwargs: Any,
    ):
        """
        A function to monitor the execution of the DARTS algorithm.

        Arguments:
            network: The DARTS network containing the weights each operation
                in the mixture architecture
            architect: The architect object used to construct the mixture architecture.
            epoch: The current epoch of the training.
            **kwargs: other parameters which may be passed from the DARTS optimizer
        """

        # collect data for visualization
        self.epoch_history.append(epoch)
        self.arch_weight_history.append(
            network.arch_parameters()[0].detach().numpy().copy()[np.newaxis, :]
        )
        self.loss_history.append(architect.current_loss)
        self.primitives = network.primitives

    def display(self):
        """
        A function to display the execution monitor. This function will generate two plots:
        (1) A plot of the training loss vs. epoch,
        (2) a plot of the architecture weights vs. epoch, divided into subplots by each edge
        in the mixture architecture.
        """

        loss_fig, loss_ax = plt.subplots(1, 1)
        loss_ax.plot(self.loss_history)

        loss_ax.set_ylabel("Loss", fontsize=14)
        loss_ax.set_xlabel("Epoch", fontsize=14)
        loss_ax.set_title("Training Loss")

        arch_weight_history_array = np.vstack(self.arch_weight_history)
        num_epochs, num_edges, num_primitives = arch_weight_history_array.shape

        subplots_per_side = int(np.ceil(np.sqrt(num_edges)))

        arch_fig, arch_axes = plt.subplots(
            subplots_per_side,
            subplots_per_side,
            sharex=True,
            sharey=True,
            figsize=(10, 10),
            squeeze=False,
        )

        arch_fig.suptitle("Architecture Weights", fontsize=10)

        for (edge_i, ax) in zip(range(num_edges), arch_axes.flat):
            for primitive_i in range(num_primitives):
                print(f"{edge_i}, {primitive_i}, {ax}")
                ax.plot(
                    arch_weight_history_array[:, edge_i, primitive_i],
                    label=f"{self.primitives[primitive_i]}",
                )

            ax.set_title("k{}".format(edge_i), fontsize=8)

            # there is no need to have the legend for each subplot
            if edge_i == 0:
                ax.legend(loc="upper center")
                ax.set_ylabel("Edge Weights", fontsize=8)
                ax.set_xlabel("Epoch", fontsize=8)

        return SimpleNamespace(
            loss_fig=loss_fig,
            loss_ax=loss_ax,
            arch_fig=arch_fig,
            arch_axes=arch_axes,
        )
