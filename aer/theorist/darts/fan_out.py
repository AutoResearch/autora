import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Fan_Out(nn.Module):

    def __init__(self, num_inputs):
        super(Fan_Out, self).__init__()

        self._num_inputs = num_inputs

        self.input_output = list()
        for i in range(num_inputs):
            linearConnection = nn.Linear(num_inputs, 1, bias=False)
            linearConnection.weight.data = torch.zeros(1,num_inputs)
            linearConnection.weight.data[0,i] = 1
            linearConnection.weight.requires_grad = False
            self.input_output.append(linearConnection)

    def forward(self, input):

        output = list()
        for i in range(self._num_inputs):
            output.append(self.input_output[i](input))

        return output

