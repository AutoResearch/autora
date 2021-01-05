from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import torch
import torch.nn as nn
from torch.autograd import Variable

class Stroop_Model(nn.Module):
    def __init__(self):
        super(Stroop_Model, self).__init__()

        # define non-learnable bias for hidden and output layers
        self.bias = Variable(torch.ones(1) * -4, requires_grad=False)

        # define affine transformations
        self.input_color_hidden_color = nn.Linear(2, 2, bias=False)
        self.input_word_hidden_word = nn.Linear(2, 2, bias=False)
        self.hidden_color_output = nn.Linear(2, 2, bias=False)
        self.hidden_word_output = nn.Linear(2, 2, bias=False)
        self.task_hidden_color = nn.Linear(2, 2, bias=False)
        self.task_hidden_word = nn.Linear(2, 2, bias=False)

        # assign weights
        self.input_color_hidden_color.weight.data = torch.FloatTensor([[2.2, -2.2], [-2.2, 2.2]])
        self.input_word_hidden_word.weight.data = torch.FloatTensor([[2.6, -2.6], [-2.6, 2.6]])
        self.hidden_color_output.weight.data = torch.FloatTensor([[1.3, -1.3], [-1.3, 1.3]])
        self.hidden_word_output.weight.data = torch.FloatTensor([[2.5, -2.5], [-2.5, 2.5]])
        self.task_hidden_color.weight.data = torch.FloatTensor([[4.0, 4.0], [0, 0]])
        self.task_hidden_word.weight.data = torch.FloatTensor([[0, 0], [4.0, 4.0]])

    def forward(self, color_red, color_green,
                      word_red, word_green,
                      task_color, task_word):

        # convert inputs
        color = torch.zeros(1, 2)
        word = torch.zeros(1, 2)
        task = torch.zeros(1, 2)

        color[0, 0] = color_red
        color[0, 1] = color_green
        word[0, 0] = word_red
        word[0, 1] = word_green
        task[0, 0] = task_color
        task[0, 1] = task_word

        color_hidden = torch.sigmoid(self.input_color_hidden_color(color) +
                                     self.task_hidden_color(task) +
                                     self.bias)

        word_hidden = torch.sigmoid(self.input_word_hidden_word(word) +
                                    self.task_hidden_word(task) +
                                    self.bias)

        output = torch.sigmoid(self.hidden_color_output(color_hidden) +
                               self.hidden_word_output(word_hidden))

        return output

class Participant_Stroop(Participant_In_Silico):

    color_red = torch.zeros(1, 1)
    color_green = torch.zeros(1, 1)
    word_red = torch.zeros(1, 1)
    word_green =  torch.zeros(1, 1)
    task_color = torch.zeros(1, 1)
    task_word = torch.zeros(1, 1)

    output = torch.zeros(1, 2)
    model = Stroop_Model()

    sample_output = False

    # initializes participant
    def __init__(self):
        super(Participant_Stroop, self).__init__()

    # read value from participant
    def get_value(self, variable_label):

        if variable_label is "verbal_red":
            return self.output[0, 0].numpy()

        if variable_label is "verbal_green":
            return self.output[0, 1].numpy()

        if variable_label is "verbal":
            return self.output.numpy()

    # assign value to participant
    def set_value(self, variable_label, value):

        if variable_label is "color_red":
            self.color_red[0, 0] = value

        if variable_label is "color_green":
            self.color_green[0, 0] = value

        if variable_label is "word_red":
            self.word_red[0, 0] = value

        if variable_label is "word_green":
            self.word_green[0, 0] = value

        if variable_label is "task_color":
            self.task_color[0, 0] = value

        if variable_label is "task_word":
            self.task_red[0, 0] = value

    def execute(self):
        output = self.model(self.color_red, self.color_green,
                            self.word_red, self.word_green,
                            self.task_color, self.task_word)

        if self.sample_output:
            probabilities = torch.exp(output) / torch.sum(torch.exp(output))
            probabilities_transformed = torch.flatten(torch.transpose(probabilities, 0, 1))
            transform_category = torch.distributions.categorical.Categorical(probabilities_transformed)
            index = transform_category.sample()
            output = torch.zeros(1, 2)
            output[index] = 1

def run_stroop_exp(model):

    # color naming - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 1
    task_word = 0
    output_col_cong = model(color_red, color_green,
                       word_red, word_green,
                       task_color, task_word)

    # color naming - incong
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 1
    task_color = 1
    task_word = 0
    output_col_incong = model(color_red, color_green,
                       word_red, word_green,
                       task_color, task_word)

    # color naming - control
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 0
    task_color = 1
    task_word = 0
    output_col_control = model(color_red, color_green,
                            word_red, word_green,
                            task_color, task_word)

    # word reading - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    output_wrd_cong = model(color_red, color_green,
                       word_red, word_green,
                       task_color, task_word)

    # word reading - incong
    color_red = 0
    color_green = 1
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    output_wrd_incong = model(color_red, color_green,
                       word_red, word_green,
                       task_color, task_word)

    # word reading - control
    color_red = 0
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    output_wrd_control = model(color_red, color_green,
                            word_red, word_green,
                            task_color, task_word)


    return (output_col_cong, output_col_incong, output_col_control,
            output_wrd_cong, output_wrd_incong, output_wrd_control)


def plot_demand_effect(model):

    # run Stroop experiment
    (output_col_cong, output_col_incong, output_col_control,
     output_wrd_cong, output_wrd_incong, output_wrd_control) = run_stroop_exp(model)

    beta = 2

    # compute accuracies
    acc_col_cong = torch.exp(output_col_cong[0, 0] * beta) / torch.sum(torch.exp(output_col_cong * beta))
    acc_col_incong = torch.exp(output_col_incong[0, 0] * beta) / torch.sum(torch.exp(output_col_incong * beta))
    acc_col_control = torch.exp(output_col_control[0, 0] * beta) / torch.sum(torch.exp(output_col_control * beta))
    acc_wrd_cong = torch.exp(output_wrd_cong[0, 0] * beta) / torch.sum(torch.exp(output_wrd_cong * beta))
    acc_wrd_incong = torch.exp(output_wrd_incong[0, 0] * beta) / torch.sum(torch.exp(output_wrd_incong * beta))
    acc_wrd_control = torch.exp(output_wrd_control[0, 0] * beta) / torch.sum(torch.exp(output_wrd_control * beta))

    err_col_cong = 1 - acc_col_cong
    err_col_incong = 1 - acc_col_incong
    err_col_control = 1 - acc_col_control
    err_wrd_cong = 1 - acc_wrd_cong
    err_wrd_incong = 1 - acc_wrd_incong
    err_wrd_control = 1 - acc_wrd_control

    # collect plot data
    x_data = [0, 1, 2]
    y_data_col = [err_col_control.detach().numpy() * 100, err_col_incong.detach().numpy() * 100, err_col_cong.detach().numpy() * 100]
    y_data_wrd = [err_wrd_control.detach().numpy() * 100, err_wrd_incong.detach().numpy() * 100, err_wrd_cong.detach().numpy() * 100]
    x_limit = [-0.5, 2.5]
    y_limit = [0, 100]
    x_label = "Condition"
    y_label = "Error Rate (%)"
    legend = ('color naming', 'word reading')

    # plot
    import matplotlib.pyplot as plt
    plt.plot(x_data, y_data_col, label=legend[0])
    plt.plot(x_data, y_data_wrd, '--', label=legend[1])
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="large")
    plt.show()

# model = Stroop_Model()
# plot_demand_effect(model)