import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class StroopNet(nn.Module):

    def __init__(self):
        super(StroopNet, self).__init__()

        # parameters are chosen according to:
        # https://github.com/qihongl/stroop-cohen-etal-1990/blob/master/stroop_model.py

        # define affine transformations (weights)
        self.color_input_hidden_color = nn.Linear(2, 2, bias=False)
        self.word_input_hidden_word = nn.Linear(2, 2, bias=False)
        self.task_input_hidden_color = nn.Linear(2, 2, bias=False)
        self.task_input_hidden_word = nn.Linear(2, 2, bias=False)
        self.color_hidden_output = nn.Linear(2, 2, bias=False)
        self.word_hidden_output = nn.Linear(2, 2, bias=False)

        # define biases
        self.hidden_bias = -4

        # assign weights
        self.color_input_hidden_color.weight.data = torch.FloatTensor([[2.2, -2.2], [-2.2, 2.2]])
        self.word_input_hidden_word.weight.data = torch.FloatTensor([[2.6, -2.6], [-2.6, 2.6]])
        self.task_input_hidden_color.weight.data = torch.FloatTensor([[4.0, 0], [4.0, 0]])
        self.task_input_hidden_word.weight.data = torch.FloatTensor([[0, 4.0], [0, 4.0]])
        self.color_hidden_output.weight.data = torch.FloatTensor([[1.3, -1.3], [-1.3, 1.3]])
        self.word_hidden_output.weight.data = torch.FloatTensor([[2.5, -2.5], [-2.5, 2.5]])

        self.logsoftmax = nn.LogSoftmax()

    def forward(self, color, word, task):

        color_hidden = torch.sigmoid(
            self.color_input_hidden_color(color) + self.task_input_hidden_color(task) + self.hidden_bias)

        word_hidden = torch.sigmoid(
            self.word_input_hidden_word(word) + self.task_input_hidden_word(task) + self.hidden_bias)

        output = torch.nn.functional.softmax(
            self.color_hidden_output(color_hidden) + self.word_hidden_output(word_hidden),
            dim = 1)

        return output

