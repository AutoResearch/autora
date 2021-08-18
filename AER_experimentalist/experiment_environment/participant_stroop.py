from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from graphviz import Digraph

class Stroop_Model(nn.Module):
    def __init__(self):
        super(Stroop_Model, self).__init__()

        # define affine transformations
        self.input_color_hidden_color = nn.Linear(2, 2, bias=False)
        self.input_word_hidden_word = nn.Linear(2, 2, bias=False)
        self.hidden_color_output = nn.Linear(2, 2, bias=False)
        self.hidden_word_output = nn.Linear(2, 2, bias=False)
        self.task_hidden_color = nn.Linear(2, 2, bias=False)
        self.task_hidden_word = nn.Linear(2, 2, bias=False)

        # assign weights
        # self.bias = Variable(torch.ones(1) * -4, requires_grad=False)
        # self.input_color_hidden_color.weight.data = torch.FloatTensor([[2.2, -2.2], [-2.2, 2.2]])
        # self.input_word_hidden_word.weight.data = torch.FloatTensor([[2.6, -2.6], [-2.6, 2.6]])
        # self.hidden_color_output.weight.data = torch.FloatTensor([[1.3, -1.3], [-1.3, 1.3]])
        # self.hidden_word_output.weight.data = torch.FloatTensor([[2.5, -2.5], [-2.5, 2.5]])
        # self.task_hidden_color.weight.data = torch.FloatTensor([[4.0, 0.0], [4.0, 0]])
        # self.task_hidden_word.weight.data = torch.FloatTensor([[0, 4.00], [0, 4.0]])

        # CONTROL STUDY
        # assign without word reading task unit
        # self.bias = Variable(torch.ones(1) * -4, requires_grad=False)
        # self.input_color_hidden_color.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.2
        # self.input_word_hidden_word.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.6
        # self.hidden_color_output.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 1.3
        # self.hidden_word_output.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5
        #
        # self.task_hidden_color.weight.data = torch.FloatTensor([[1.0, 0.0], [1.0, 0]]) * 4
        # self.task_hidden_word.weight.data = torch.FloatTensor([[0, 1], [0, 1]]) * 0

        self.bias = Variable(torch.ones(1) * -4, requires_grad=False)
        self.input_color_hidden_color.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5
        self.hidden_color_output.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5

        self.input_word_hidden_word.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.6
        self.hidden_word_output.weight.data = torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5

        self.task_hidden_color.weight.data = torch.FloatTensor([[1.0, 0.0], [1.0, 0]]) * 4
        self.task_hidden_word.weight.data = torch.FloatTensor([[0, 1], [0, 1]]) * 0

    def forward(self, input):

        input = torch.Tensor(input)
        if len(input.shape) <= 1:
            input = input.view(1, len(input))

        # convert inputs
        color = torch.zeros(input.shape[0], 2)
        word = torch.zeros(input.shape[0], 2)
        task = torch.zeros(input.shape[0], 2)

        color[:, 0:2] = input[:, 0:2]
        word[:, 0:2] = input[:, 2:4]
        task[:, 0:2] = input[:, 4:6]

        color_hidden = torch.sigmoid(self.input_color_hidden_color(color) +
                                     self.task_hidden_color(task) +
                                     self.bias)

        word_hidden = torch.sigmoid(self.input_word_hidden_word(word) +
                                    self.task_hidden_word(task) +
                                    self.bias)

        output = self.hidden_color_output(color_hidden) + \
                 self.hidden_word_output(word_hidden)

        return output

class Participant_Stroop(Participant_In_Silico):

    # initializes participant
    def __init__(self):
        super(Participant_Stroop, self).__init__()

        self.color_red = torch.zeros(1, 1)
        self.color_green = torch.zeros(1, 1)
        self.word_red = torch.zeros(1, 1)
        self.word_green = torch.zeros(1, 1)
        self.task_color = torch.ones(1, 1)  # color task units are activated by default
        self.task_word = torch.zeros(1, 1)

        self.output = torch.zeros(1, 2)
        self.output_sample = 0
        self.model = Stroop_Model()

    # read value from participant
    def get_value(self, variable_name):

        if variable_name is "verbal_red":
            return self.output[0, 0].numpy()

        elif variable_name is "verbal_green":
            return self.output[0, 1].numpy()

        elif variable_name is "verbal_sample":
            return self.output_sample.numpy()

        elif variable_name is "verbal":
            return self.output.numpy()

        raise Exception('Could not get value from Stroop Participant. Variable name "' + variable_name + '" not found.')

    # assign value to participant
    def set_value(self, variable_name, value):

        if variable_name is "color_red":
            self.color_red[0, 0] = value

        elif variable_name is "color_green":
            self.color_green[0, 0] = value

        elif variable_name is "word_red":
            self.word_red[0, 0] = value

        elif variable_name is "word_green":
            self.word_green[0, 0] = value

        elif variable_name is "task_color":
            self.task_color[0, 0] = value

        elif variable_name is "task_word":
            self.task_word[0, 0] = value

        else:
            raise Exception('Could not set value for Stroop Participant. Variable name "' + variable_name + '" not found.')

    def execute(self):

        input = torch.zeros(1, 6)
        input[0, 0] = self.color_red
        input[0, 1] = self.color_green
        input[0, 2] = self.word_red
        input[0, 3] = self.word_green
        input[0, 4] = self.task_color
        input[0, 5] = self.task_word

        # compute regular output
        output_net = self.model(input).detach()
        self.output = torch.sigmoid(output_net)

        # compute sample from softmax
        probabilities = torch.exp(output_net) / torch.sum(torch.exp(output_net))
        probabilities_transformed = torch.flatten(torch.transpose(probabilities, 0, 1))
        transform_category = torch.distributions.categorical.Categorical(probabilities_transformed)
        index = transform_category.sample()
        self.output_sample = index

    def compute_BIC(self, object_of_study, num_params = None):

        (input, target) = object_of_study.get_dataset()

        input_full = torch.zeros(input.shape[0], 6)
        input_full[:, 0] = self.color_red[0, 0]
        input_full[:, 1] = self.color_green[0, 0]
        input_full[:, 2] = self.word_red[0, 0]
        input_full[:, 3] = self.word_green[0, 0]
        input_full[:, 4] = self.task_color[0, 0]
        input_full[:, 5] = self.task_word[0, 0]

        for idx, IV in enumerate(object_of_study.independent_variables):
            variable_name = IV.get_name()
            if variable_name is "color_red":
                input_full[:, 0] = input[:, idx]
            if variable_name is "color_green":
                input_full[:, 1] = input[:, idx]
            if variable_name is "word_red":
                input_full[:, 2] = input[:, idx]
            if variable_name is "word_green":
                input_full[:, 3] = input[:, idx]
            if variable_name is "task_color":
                input_full[:, 4] = input[:, idx]
            if variable_name is "task_word":
                input_full[:, 5] = input[:, idx]

        output_fnc = nn.Softmax(dim=1)
        return super(Participant_Stroop, self).compute_BIC(input_full, target, output_fnc, num_params)

    def graph_simple(self, filepath):

        # formatting
        decimals = 2
        format_string = "{:." + "{:.0f}".format(decimals) + "f}"

        # set up graph
        g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times"),
            node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                           penwidth='2', fontname="times"),
            engine='dot')
        g.body.extend(['rankdir=LR'])

        # add input nodes
        red = 'Color Red'
        green = 'Color Green'
        g.node('Color Red', fillcolor='#F1EDB9')
        g.node('Color Green', fillcolor='#F1EDB9')

        # add hidden nodes
        hidden1 = '0'
        hidden2 = '1'
        g.node('0', fillcolor='#BBCCF9')
        g.node('1', fillcolor='#BBCCF9')

        # add output node
        out1 = 'Logistic(x1)'
        out2 = 'Logistic(x2)'
        g.node(out1, fillcolor='#CBE7C7')
        g.node(out2, fillcolor='#CBE7C7')

        # add links from input to hidden
        value = self.model.input_color_hidden_color.weight.data[0, 0]
        bias = self.model.bias[0]
        str = format_string.format(value) + " * x " + format_string.format(bias)
        g.edge(red, hidden1, label=str, fillcolor="gray")

        value = self.model.input_color_hidden_color.weight.data[0, 1]
        str = format_string.format(value) + " * x"
        g.edge(red, hidden2, label=str, fillcolor="gray")

        value = self.model.input_color_hidden_color.weight.data[1, 0]
        str = format_string.format(value) + " * x"
        g.edge(green, hidden1, label=str, fillcolor="gray")

        value = self.model.input_color_hidden_color.weight.data[1, 1]
        str = format_string.format(value) + " * x " + format_string.format(bias)
        g.edge(green, hidden2, label=str, fillcolor="gray")

        # add links from hidden to output
        value1 = self.model.hidden_color_output.weight.data[0, 0]
        value2 = self.model.hidden_color_output.weight.data[0, 1]
        # str = 'x.*(' + format_string.format(value1) + ' + ' + format_string.format(value2) + ')'
        str = format_string.format(value1) + " * x " + format_string.format(bias)
        g.edge(hidden1, out1, label=str, fillcolor="gray")
        str = format_string.format(value2) + " * x"
        g.edge(hidden1, out2, label=str, fillcolor="gray")

        value1 = self.model.hidden_color_output.weight.data[1, 0]
        value2 = self.model.hidden_color_output.weight.data[1, 1]
        str = format_string.format(value1) + " * x"
        g.edge(hidden2, out1, label=str, fillcolor="gray")
        str = format_string.format(value2) + " * x " + format_string.format(bias)
        g.edge(hidden2, out2, label=str, fillcolor="gray")

        # save graph
        g.render(filepath, view=False)

    def figure_control_plot(self, comparison_model,
                    color_green_list=(0, 0, 1),
                    task_color_list=(0, 1, 1),
                    num_data_points=100,
                    figures_path=None,
                    figure_name=None,
                    figure_dimensions=(4, 3),
                    y_limit=[0, 1],
                    legend_font_size=8,
                    axis_font_size=10,
                    title_font_size=10):

        ground_truth = self.model
        approximation = comparison_model

        output_truth = run_control_exp(ground_truth, color_green_list, task_color_list, num_data_points)

        output_approx = run_control_exp(approximation, color_green_list, task_color_list, num_data_points, input_dim=3)

        # collect plot data
        x_data = np.linspace(0, 1, num_data_points)
        y1_truth_data = output_truth[0]
        y2_truth_data = output_truth[1]
        y3_truth_data = output_truth[2]
        y1_approx_data = output_approx[0]
        y2_approx_data = output_approx[1]
        y3_approx_data = output_approx[2]

        x_limit = [0, 1]
        x_label = "Red Input"
        y_label = 'Red Response'
        legend = list()
        for color_green, task_color in zip(color_green_list, task_color_list):
            legend.append('$act_{green}$ = ' + str(color_green) + ', $act_{task}$ = ' + str(task_color) + ' (GT)')
        for color_green, task_color in zip(color_green_list, task_color_list):
            legend.append('$act_{green}$ = ' + str(color_green) + ', $act_{task}$ = ' + str(task_color) + ' (R)')

        # plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import pyplot
        import os

        fig, ax = pyplot.subplots(figsize=figure_dimensions)

        ax.plot(x_data, y1_truth_data, '-', label=legend[0], color='#CC6677')
        ax.plot(x_data, y2_truth_data, '-', label=legend[1], color='#44AA99')
        ax.plot(x_data, y3_truth_data, '-', label=legend[2], color='#332288')
        ax.plot(x_data, y1_approx_data, '--', label=legend[3], color='#CC6677')
        ax.plot(x_data, y2_approx_data, '--', label=legend[4], color='#44AA99')
        ax.plot(x_data, y3_approx_data, '--', label=legend[5], color='#332288')
        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        ax.set_xlabel(x_label, fontsize=axis_font_size)
        ax.set_ylabel(y_label, fontsize=axis_font_size)
        ax.set_title('Response Function', fontsize=title_font_size)
        ax.legend(loc=0, fontsize=legend_font_size, bbox_to_anchor=(1.05, 1))
        # sns.despine(trim=True)
        plt.show()

        if figure_name is not None and figures_path is not None:
            if not os.path.exists(figures_path):
                os.mkdir(figures_path)
            fig.savefig(os.path.join(figures_path, figure_name))

def run_control_exp(model, color_green_list, color_task_list, num_data_points, input_dim=None):

    output_list = list()

    for color_green, color_task in zip(color_green_list, color_task_list):
        output_list.append(list())

        color_red_list = np.linspace(0, 1, num_data_points)
        for color_red in color_red_list:
            if input_dim is None:
                input = torch.zeros(1, 6)
                input[0, 0] = color_red
                input[0, 1] = color_green
                input[0, 4] = color_task
            else:
                input = torch.zeros(1, 3)
                input[0, 0] = color_red
                input[0, 1] = color_green
                input[0, 2] = color_task
            output = torch.sigmoid(model(input)).detach().numpy().flatten()[0]
            output_list[-1].append(output)

    return output_list


def run_exp(model):

    # color naming - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 1
    task_word = 0
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_col_cong = torch.sigmoid(model(input))

    # color naming - incong
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 1
    task_color = 1
    task_word = 0
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_col_incong = torch.sigmoid(model(input))

    # color naming - control
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 0
    task_color = 1
    task_word = 0
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_col_control = torch.sigmoid(model(input))

    # word reading - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_wrd_cong = torch.sigmoid(model(input))

    # word reading - incong
    color_red = 0
    color_green = 1
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_wrd_incong = torch.sigmoid(model(input))

    # word reading - control
    color_red = 0
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_red, color_green, word_red, word_green, task_color, task_word]
    output_wrd_control = torch.sigmoid(model(input))


    return (output_col_cong, output_col_incong, output_col_control,
            output_wrd_cong, output_wrd_incong, output_wrd_control)


def plot_demand_effect(model):

    # run Stroop experiment
    (output_col_cong, output_col_incong, output_col_control,
     output_wrd_cong, output_wrd_incong, output_wrd_control) = run_exp(model)

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