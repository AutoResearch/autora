from typing import Tuple, cast

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable

from autora.experimentalist.pooler.poppernet import (
    PopperNet,
    class_to_onehot,
    plot_popper_diagnostics,
)
from autora.variable import ValueType, VariableCollection


def falsification_sampler(
    X,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metadata: VariableCollection,
    n: int = 100,
    training_epochs: int = 1000,
    training_lr: float = 1e-3,
    mse_scale: float = 1,
    plot: bool = False,
):
    """
    A Sampler that generates samples for independent variables with the objective of maximizing the
    (approximated) loss of the model. The samples are generated by first training a neural network
    to approximate the loss of a model for all patterns in the training data_closed_loop.
    Once trained, the network is then provided with the candidate samples and the samples with
    the highest loss are selected.

    Args:
        X: The candidate samples to be evaluated.
        model: Scikit-learn model, could be either a classification or regression model
        x_train: data_closed_loop that the model was trained on
        y_train: labels that the model was trained on
        metadata: Meta-data_closed_loop about the dependent and independent variables
        n: number of samples to return
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        training_lr: learning rate for training the popper network
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Samples with the highest loss

    """

    # format input

    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    x_train = np.array(x_train)
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(-1, 1)

    y_train = np.array(y_train)
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    if metadata.dependent_variables[0].type == ValueType.CLASS:
        # find all unique values in y_train
        num_classes = len(np.unique(y_train))
        y_train = class_to_onehot(y_train, n_classes=num_classes)

    # create list of IV limits
    ivs = metadata.independent_variables
    iv_limit_list = list()
    for iv in ivs:
        if hasattr(iv, "value_range"):
            value_range = cast(Tuple, iv.value_range)
            lower_bound = value_range[0]
            upper_bound = value_range[1]
            iv_limit_list.append(([lower_bound, upper_bound]))

    # get dimensions of input and output
    n_input = len(metadata.independent_variables)
    n_output = len(metadata.dependent_variables)

    # get input pattern for popper net
    popper_input = Variable(torch.from_numpy(x_train), requires_grad=False).float()

    # get target pattern for popper net
    model_predict = getattr(model, "predict_proba", None)
    if callable(model_predict) is False:
        model_predict = getattr(model, "predict", None)

    if callable(model_predict) is False or model_predict is None:
        raise Exception("Model must have `predict` or `predict_proba` method.")

    model_prediction = model_predict(x_train)
    if isinstance(model_prediction, np.ndarray) is False:
        try:
            model_prediction = np.array(model_prediction)
        except Exception:
            raise Exception("Model prediction must be convertable to numpy array.")
    if model_prediction.ndim == 1:
        model_prediction = model_prediction.reshape(-1, 1)

    criterion = nn.MSELoss()
    model_loss = (model_prediction - y_train) ** 2 * mse_scale
    model_loss = np.mean(model_loss, axis=1)

    # standardize the loss
    scaler = StandardScaler()
    model_loss = scaler.fit_transform(model_loss.reshape(-1, 1)).flatten()

    model_loss = torch.from_numpy(model_loss).float()
    popper_target = Variable(model_loss, requires_grad=False)

    # create the network
    popper_net = PopperNet(n_input, n_output)

    # reformat input in case it is 1D
    if len(popper_input.shape) == 1:
        popper_input = popper_input.flatten()
        popper_input = popper_input.reshape(-1, 1)

    # define the optimizer
    popper_optimizer = torch.optim.Adam(popper_net.parameters(), lr=training_lr)

    # train the network
    losses = []
    for epoch in range(training_epochs):
        popper_prediction = popper_net(popper_input)
        loss = criterion(popper_prediction, popper_target.reshape(-1, 1))
        popper_optimizer.zero_grad()
        loss.backward()
        popper_optimizer.step()
        losses.append(loss.item())

    if plot:
        popper_input_full = np.linspace(
            iv_limit_list[0][0], iv_limit_list[0][1], 1000
        ).reshape(-1, 1)
        popper_input_full = Variable(
            torch.from_numpy(popper_input_full), requires_grad=False
        ).float()
        popper_prediction = popper_net(popper_input_full)
        plot_popper_diagnostics(
            losses,
            popper_input,
            popper_input_full,
            popper_prediction,
            popper_target,
            model_prediction,
            y_train,
        )

    # now that the popper network is trained we can assign losses to all data points to be evaluated
    popper_input = Variable(torch.from_numpy(X), requires_grad=True).float()
    Y = popper_net(popper_input).detach().numpy().flatten()

    # order rows in Y from highest to lowest
    sorted_X = X[np.argsort(Y)[::-1]]

    return sorted_X[:n]
