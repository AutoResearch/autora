from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np

class Weber_Model(nn.Module):
    def __init__(self, k=1, amplification=1):
        super(Weber_Model, self).__init__()
        self.k = k
        self.amplification = amplification

    def forward(self, input):

        input = torch.Tensor(input)
        if len(input.shape) <= 1:
            input = input.view(1, len(input))

        # convert inputs
        S1 = torch.zeros(1, 1)
        S2 = torch.zeros(1, 1)

        S1[:, 0] = input[:, 0]
        S2[:, 0] = input[:, 1]

        difference = S2-S1
        JND = self.k * S1

        output = self.amplification*(difference-JND)

        return output

class Participant_Weber(Participant_In_Silico):

    # initializes participant
    def __init__(self, k=1):
        super(Participant_Weber, self).__init__()
        self.model = Weber_Model(k=k)

        self.S1 = torch.zeros(1, 1)
        self.S2 = torch.zeros(1, 1)

        self.output = torch.zeros(1, 1)
        self.output_sampled = torch.zeros(1, 1)

    # read value from participant
    def get_value(self, variable_name):

        if variable_name is "difference_detected":
            return self.output[0, 0].numpy()

        elif variable_name is "difference_detected_sample":
            return self.output_sampled[0, 0].numpy()

        raise Exception('Could not get value from Weber Participant. Variable name "' + variable_name + '" not found.')

    # assign value to participant
    def set_value(self, variable_name, value):

        if variable_name is "S1":
            self.S1[0, 0] = value

        elif variable_name is "S2":
            self.S2[0, 0] = value

        else:
            raise Exception('Could not set value for Weber Participant. Variable name "' + variable_name + '" not found.')

    def execute(self):

        input = torch.zeros(1, 6)
        input[0, 0] = self.S1
        input[0, 1] = self.S2

        # compute regular output
        output_net = self.model(input).detach()
        self.output = torch.sigmoid(output_net)
        sample = random.uniform(0, 1)
        if sample < self.output:
            self.output_sampled[0] = 1
        else:
            self.output_sampled[0] = 0

def run_exp(model, S1_list, max_diff, num_data_points=100):



    diff1 = list()
    output1 = list()

    diff2 = list()
    output2 = list()

    diff3 = list()
    output3 = list()

    S1 = S1_list[0]
    S2_list = np.linspace(S1, S1+max_diff, num_data_points)
    for S2 in S2_list:
        input = [S1, S2]
        output = torch.sigmoid(model(input))
        diff1.append(S2-S1)
        output1.append(output)

    S1 = S1_list[1]
    S2_list = np.linspace(S1, S1 + max_diff, num_data_points)
    for S2 in S2_list:
        input = [S1, S2]
        output = torch.sigmoid(model(input))
        diff2.append(S2 - S1)
        output2.append(output)

    S1 = S1_list[2]
    S2_list = np.linspace(S1, S1 + max_diff, num_data_points)
    for S2 in S2_list:
        input = [S1, S2]
        output = torch.sigmoid(model(input))
        diff3.append(S2 - S1)
        output3.append(output)

    return (diff1, diff2, diff3,
            output1, output2, output3)


def plot_psychophysics(model, S1_list=(1, 2.5, 4), max_diff=5, num_data_points=100):

    (diff1, diff2, diff3,
     output1, output2, output3) = run_exp(model, S1_list, max_diff, num_data_points)

    # collect plot data
    x1_data = diff1
    x2_data = diff2
    x3_data = diff3
    y1_data = output1
    y2_data = output2
    y3_data = output3

    x_limit = [0, max_diff]
    y_limit = [0, 1]
    x_label = "S2-S1"
    y_label = "P(Detected)"
    legend = list()
    for S1 in S1_list:
        legend.append('S1 = ' + str(S1))

    # plot
    import matplotlib.pyplot as plt
    plt.plot(x1_data, y1_data, label=legend[0])
    plt.plot(x2_data, y2_data, '--', label=legend[1])
    plt.plot(x3_data, y3_data, '.-', label=legend[2])
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="large")
    plt.show()

# model = Weber_Model()
# plot_psychophysics(model)