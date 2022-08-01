import torch
import torch.nn as nn


class Fan_Out(nn.Module):
    """
    A neural network class that splits a given input vector into separate nodes. Each element of
    the original input vector is allocated a separate node in a computation graph.
    """

    def __init__(self, num_inputs: int):
        """
        Initialize the Fan Out operation.

        Arguments:
                num_inputs (int): The number of distinct input nodes to generate
        """
        super(Fan_Out, self).__init__()

        self._num_inputs = num_inputs

        self.input_output = list()
        for i in range(num_inputs):
            linearConnection = nn.Linear(num_inputs, 1, bias=False)
            linearConnection.weight.data = torch.zeros(1, num_inputs)
            linearConnection.weight.data[0, i] = 1
            linearConnection.weight.requires_grad = False
            self.input_output.append(linearConnection)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Fan Out operation.

        Arguments:
            input: input vector whose elements are split into separate input nodes
        """

        output = list()
        for i in range(self._num_inputs):
            output.append(self.input_output[i](input))

        return output
