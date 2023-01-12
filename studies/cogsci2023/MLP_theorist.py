from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.autograd import Variable
from autora.variable import ValueType
import torch
from typing import Tuple
from matplotlib import pyplot as plt

class MLP_theorist(BaseEstimator, RegressorMixin):

    def __init__(self, epochs=1000, lr=1e-4, n_hidden: Tuple = (32, 16, 32),
                 output_type=ValueType.REAL, seed=0, verbose=False):
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.output_type = output_type
        self.verbose = verbose
        self.network = None

    def fit(self, X, y):

        # create multi-layer perceptron
        network = MLP(X.shape[1], y.shape[1],
                      output_type=self.output_type,
                      n_hidden=self.n_hidden)

        if self.output_type == ValueType.REAL:
            criterion = nn.MSELoss()
        elif self.output_type == ValueType.PROBABILITY:
            criterion = nn.MSELoss()
        elif self.output_type == ValueType.CLASS:
            criterion = nn.CrossEntropyLoss()
        elif self.output_type == ValueType.PROBABILITY_SAMPLE:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        input = Variable(torch.from_numpy(X), requires_grad=False).float()
        target = Variable(torch.from_numpy(y), requires_grad=False).float()

        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
        # cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        losses = []
        for epoch in range(self.epochs):
            prediction = network(input)
            loss = criterion(prediction, target)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: {loss.item()}')

        if self.verbose:
            # plot loss over time
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()

        self.network = network

        return self

    def predict(self, X):

        input = Variable(torch.from_numpy(X), requires_grad=False).float()

        y = self.network(input).detach().numpy()
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        return y


# define the network
class MLP(nn.Module):
    def __init__(self, n_input: torch.Tensor, n_output: torch.Tensor,
                 output_type: ValueType = ValueType.REAL,
                 n_hidden: Tuple[int, int, int] = (64, 64, 64)):
        # Perform initialization of the pytorch superclass
        super(MLP, self).__init__()

        # Define network layer dimensions
        D_in, H1, H2, H3, D_out = [n_input, n_hidden[0], n_hidden[1], n_hidden[2], n_output]

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

        self.output_type = output_type
        self.n_input = n_input
        self.n_output = n_output

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

        if self.output_type == ValueType.REAL:
            return x
        elif self.output_type == ValueType.PROBABILITY and self.n_output == 1:
            return torch.sigmoid(x)
        elif self.output_type == ValueType.PROBABILITY and self.n_output > 1:
            return torch.softmax(x, dim=1)
        elif self.output_type == ValueType.PROBABILITY_SAMPLE:
            return torch.softmax(x, dim=1)
        else:
            return x
