import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        # # define affine transformations
        self.net_stimulus1Hidden1 = nn.Linear(1, 1, bias=False)
        self.net_stimulus2Hidden2 = nn.Linear(1, 1, bias=False)
        self.net_hidden1Output = nn.Linear(1, 2, bias=False)
        self.net_hidden2Output = nn.Linear(1, 2, bias=False)

        # assign weights
        self.net_stimulus1Hidden1.weight.data = torch.eye(1) * 3 # 4
        self.net_stimulus2Hidden2.weight.data = torch.eye(1) * 0
        self.net_hidden1Output.weight.data[0, 0] = 1
        self.net_hidden1Output.weight.data[1, 0] = 0
        self.net_hidden2Output.weight.data[0, 0] = 0
        self.net_hidden2Output.weight.data[1, 0] = 1

        self.logsoftmax = nn.LogSoftmax()

    def forward(self, stimulus1, stimulus2):

        # HIDDEN (0)

        # hidden1 = self.net_stimulus1Hidden1(stimulus1)                        # (0) LINEAR
        # hidden1 = torch.sigmoid(self.net_stimulus1Hidden1(stimulus1))         # (0) SIGMOID
        hidden1 = torch.exp(self.net_stimulus1Hidden1(stimulus1))                 # (0) EXP
        # hidden1 = torch.nn.functional.relu(self.net_stimulus1Hidden1(stimulus1))  # (0) RELU

        # HIDDEN (1)

        hidden2 = self.net_stimulus2Hidden2(stimulus2)                        # (1) LINEAR
        # hidden2 = torch.sigmoid(self.net_stimulus2Hidden2(stimulus1))         # (1) SIGMOID
        # hidden2 = torch.exp(self.net_stimulus2Hidden2(stimulus1))              # (1) EXP
        # hidden2 = torch.nn.functional.relu(self.net_stimulus2Hidden2(stimulus1))  # (1) RELU


        output_net = self.net_hidden1Output(hidden1) + self.net_hidden2Output(hidden2)
        output  = torch.nn.functional.softmax(output_net, dim=1)

        return output

