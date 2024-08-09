from typing import List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable

from autora.variable import VariableCollection

from .utils import plot_falsification_diagnostics


# define the network
class PopperNet(nn.Module):
    def __init__(self, n_input: torch.Tensor, n_output: torch.Tensor):
        # Perform initialization of the pytorch superclass
        super(PopperNet, self).__init__()

        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [n_input, 64, 64, 64, n_output]

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x: torch.Tensor):
        """
        This method defines the network layering and activation functions
        """
        x = self.linear1(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear2(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear3(x)  # hidden layer
        x = torch.tanh(x)  # activation function

        x = self.linear4(x)  # output layer

        return x

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False


def train_popper_net(
    model_prediction,
    reference_conditions: np.ndarray,
    reference_observations: np.ndarray,
    metadata: VariableCollection,
    iv_limit_list: List,
    training_epochs: int = 1000,
    training_lr: float = 1e-3,
    plot: bool = False,
):
    """
    Trains a neural network to approximate the loss of a model for all patterns in the training data
    Once trained, the network is then inverted to generate samples that maximize the approximated
    loss of the model.

    Note: If the pooler returns samples that are close to the boundaries of the variable space,
    then it is advisable to increase the limit_repulsion parameter (e.g., to 0.000001).

    Args:
        model: Scikit-learn model, could be either a classification or regression model
        reference_conditions: data that the model was trained on
        reference_observations: labels that the model was trained on
        metadata: Meta-data about the dependent and independent variables
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        training_lr: learning rate for training the popper network
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Trained popper net.

    """

    # get dimensions of input and output
    n_input = reference_conditions.shape[1]
    n_output = 1  # only predicting one MSE

    # get input pattern for popper net
    popper_input = Variable(
        torch.from_numpy(reference_conditions), requires_grad=False
    ).float()

    # get target pattern for popper net
    if isinstance(model_prediction, np.ndarray) is False:
        try:
            model_prediction = np.array(model_prediction)
        except Exception:
            raise Exception("Model prediction must be convertable to numpy array.")
    if model_prediction.ndim == 1:
        model_prediction = model_prediction.reshape(-1, 1)

    criterion = nn.MSELoss()
    model_loss = (model_prediction - reference_observations) ** 2
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
        if len(iv_limit_list) > 1:
            Warning(
                "Plotting currently not supported for more than two independent variables."
            )
        else:
            popper_input_full = np.linspace(
                iv_limit_list[0][0], iv_limit_list[0][1], 1000
            ).reshape(-1, 1)
            popper_input_full = Variable(
                torch.from_numpy(popper_input_full), requires_grad=False
            ).float()
            popper_prediction = popper_net(popper_input_full)
            plot_falsification_diagnostics(
                losses,
                popper_input,
                popper_input_full,
                popper_prediction,
                popper_target,
                model_prediction,
                reference_observations,
            )

    return popper_net, model_loss


def train_popper_net_with_model(
    model,
    reference_conditions: np.ndarray,
    reference_observations: np.ndarray,
    metadata: VariableCollection,
    iv_limit_list: List,
    training_epochs: int = 1000,
    training_lr: float = 1e-3,
    plot: bool = False,
):
    """
    Trains a neural network to approximate the loss of a model for all patterns in the training data
    Once trained, the network is then inverted to generate samples that maximize the approximated
    loss of the model.

    Note: If the pooler returns samples that are close to the boundaries of the variable space,
    then it is advisable to increase the limit_repulsion parameter (e.g., to 0.000001).

    Args:
        model: Scikit-learn model, could be either a classification or regression model
        reference_conditions: data that the model was trained on
        reference_observations: labels that the model was trained on
        metadata: Meta-data about the dependent and independent variables
        training_epochs: number of epochs to train the popper network for approximating the
        error fo the model
        training_lr: learning rate for training the popper network
        plot: print out the prediction of the popper network as well as its training loss

    Returns: Trained popper net.

    """

    model_predict = getattr(model, "predict_proba", None)
    if callable(model_predict) is False:
        model_predict = getattr(model, "predict", None)

    if callable(model_predict) is False or model_predict is None:
        raise Exception("Model must have `predict` or `predict_proba` method.")

    model_prediction = model_predict(reference_conditions)

    return train_popper_net(
        model_prediction,
        reference_conditions,
        reference_observations,
        metadata,
        iv_limit_list,
        training_epochs,
        training_lr,
        plot,
    )
