import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class StroopNet(nn.Module):

    def __init__(self):
        super(StroopNet, self).__init__()

        # # define non-learnable bias for hidden and output layers
        self.bias = Variable(torch.ones(1) * -4, requires_grad = False)

        # # define affine transformations
        self.net_stimulusHidden = nn.Linear(4, 4, bias=False)
        self.net_taskHidden = nn.Linear(2, 4, bias=False)
        self.net_hiddenOutput = nn.Linear(4, 2, bias=False)
        self.net_taskOutput = nn.Linear(2, 2, bias=False)
        self.fc1 = nn.Linear(4, 4)

        # assign weights
        self.net_stimulusHidden.weight.data = torch.eye(4) * 2

        self.net_taskHidden.weight.data[0, :] = torch.FloatTensor([1, 0]) * 3
        self.net_taskHidden.weight.data[1, :] = torch.FloatTensor([1, 0]) * 3
        self.net_taskHidden.weight.data[2, :] = torch.FloatTensor([0, 1]) * 3
        self.net_taskHidden.weight.data[3, :] = torch.FloatTensor([0, 1]) * 3

        self.net_hiddenOutput.weight.data[0, :] = torch.FloatTensor([1, -1, 2, -2]) * 10
        self.net_hiddenOutput.weight.data[1, :] = torch.FloatTensor([-1, 1, -2, 2]) * 10

        self.net_taskOutput.weight.data = torch.ones(2,2) * 3


    def forward(self, stimulus, task):

        hidden = torch.sigmoid(self.net_stimulusHidden(stimulus) + self.net_taskHidden(task) + self.bias)
        output = torch.sigmoid(self.net_hiddenOutput(hidden) + self.net_taskOutput(task) + self.bias)

        return output


net = StroopNet()
print(net)