from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from graphviz import Digraph

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
        S1 = torch.zeros(input.shape[0], 1)
        S2 = torch.zeros(input.shape[0], 1)

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

    def compute_BIC(self, object_of_study, num_params = 0):

        (input, target) = object_of_study.get_dataset()

        input_full = torch.zeros(input.shape[0], 6)
        input_full[:, 0] = self.S1[0, 0]
        input_full[:, 1] = self.S2[0, 0]

        for idx, IV in enumerate(object_of_study.independent_variables):
            variable_name = IV.get_name()
            if variable_name is "S1":
                input_full[:, 0] = input[:, idx]
            if variable_name is "S2":
                input_full[:, 1] = input[:, idx]

        output_fnc = nn.Sigmoid()
        return super(Participant_Weber, self).compute_BIC(input_full, target, output_fnc, num_params)

    def graph_soft(self, filepath):

        # formatting
        decimals = 2
        format_string = "{:." + "{:.0f}".format(decimals) + "f}"

        # set up graph
        g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times", penwidth='3'),
            node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                           penwidth='2', fontname="times"),
            engine='dot')
        g.body.extend(['rankdir=LR'])

        # add input nodes
        input1 = 'x_0'
        input2 = 'x_1'
        g.node(input1, fillcolor='#F1EDB9')
        g.node(input2, fillcolor='#F1EDB9')

        hidden1 = 'x_2'
        hidden2 = 'x_3'
        hidden3 = 'x_4'
        g.node(hidden1, fillcolor='#BBCCF9')
        g.node(hidden2, fillcolor='#BBCCF9')
        g.node(hidden3, fillcolor='#BBCCF9')

        # add output node
        out1 = 'P(detected)'
        g.node(out1, fillcolor='#CBE7C7')
        out2 = 'r'
        g.node(out2, fillcolor='#CBE7C7')

        # add links from input to hidden
        strength1 = '1'
        strength2 = '2'
        strength3 = '3'
        strength4 = '4'

        # g.attr('edge', color="#000000", penwidth=strength3) # softmax
        # g.edge(input1, hidden1, fillcolor="#000000")
        # g.attr('edge', color="#44AA99", penwidth=strength2) # + x
        # g.edge(input1, hidden1, fillcolor="#44AA99")
        # g.attr('edge', color="#CC6677", penwidth=strength1) # - x
        # g.edge(input1, hidden1, fillcolor="#CC6677")
        # g.attr('edge', color="#332288", penwidth=strength4) # k * x
        # g.edge(input1, hidden1, fillcolor="#332288")
        #
        # g.attr('edge', color="#000000", penwidth=strength1) # softmax
        # g.edge(input1, hidden2, fillcolor="#000000")
        # g.attr('edge', color="#44AA99", penwidth=strength3) # + x
        # g.edge(input1, hidden2, fillcolor="#44AA99")
        # g.attr('edge', color="#CC6677", penwidth=strength4) # - x
        # g.edge(input1, hidden2, fillcolor="#CC6677")
        # g.attr('edge', color="#332288", penwidth=strength2) # k * x
        # g.edge(input1, hidden2, fillcolor="#332288")
        #
        # g.attr('edge', color="#000000", penwidth=strength1) # softmax
        # g.edge(input2, hidden2, fillcolor="#000000")
        # g.attr('edge', color="#44AA99", penwidth=strength4) # + x
        # g.edge(input2, hidden2, fillcolor="#44AA99")
        # g.attr('edge', color="#CC6677", penwidth=strength3) # - x
        # g.edge(input2, hidden2, fillcolor="#CC6677")
        # g.attr('edge', color="#332288", penwidth=strength2) # k * x
        # g.edge(input2, hidden2, fillcolor="#332288")

        # save graph
        g.render(filepath, view=False)


    def graph_simple(self, filepath):

        # formatting
        decimals = 2
        format_string = "{:." + "{:.0f}".format(decimals) + "f}"

        # set up graph
        g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times", penwidth='3'),
            node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                           penwidth='2', fontname="times"),
            engine='dot')
        g.body.extend(['rankdir=LR'])

        # add input nodes
        input1 = 'x_0'
        input2 = 'x_1'
        g.node(input1, fillcolor='#F1EDB9')
        g.node(input2, fillcolor='#F1EDB9')

        hidden1 = 'x_2'
        hidden2 = 'x_3'
        hidden3 = 'x_3'
        g.node(hidden1, fillcolor='#BBCCF9')
        g.node(hidden2, fillcolor='#BBCCF9')
        # g.node(hidden3, fillcolor='#BBCCF9')

        # add output node
        out1 = 'P(detected)'
        # g.node(out1, fillcolor='#CBE7C7')

        out2 = 'r'
        g.node(out2, fillcolor='#CBE7C7')

        # add links from input to hidden
        str = "0.5 * x"
        g.attr('edge', color="#4472C4")
        g.edge(input1, hidden1, label=str, fillcolor="#4472C4")

        str = "-x"
        g.attr('edge', color="#CC6677")
        g.edge(input1, hidden2, label=str, fillcolor="#CC6677")

        str = "+x"
        g.attr('edge', color="#44AA99")
        g.edge(input2, hidden2, label=str, fillcolor="#44AA99")

        str = "-1 * x"
        g.attr('edge', color="#4472C4")
        g.edge(hidden1, out2, label=str, fillcolor="#4472C4") # 332288
        str = "1 * x"
        g.attr('edge', color="#4472C4")
        g.edge(hidden2, out2, label=str, fillcolor="#4472C4")

        # str = "logistic(x)"
        # g.attr('edge', color="#000000")
        # g.edge(hidden3, out1, label=str, fillcolor="black")

        # save graph
        g.render(filepath, view=False)

    def figure_plot(self, comparison_model,
                    S1_list=(1, 2.5, 4),
                    max_diff=5,
                    num_data_points=100,
                    figures_path=None,
                    figure_name=None,
                    figure_dimensions=(4, 3),
                    y_limit= [0, 1],
                    legend_font_size=8,
                    axis_font_size=10,
                    title_font_size=10):

        ground_truth = self.model
        approximation = comparison_model

        (diff1, diff2, diff3,
         output1_truth, output2_truth, output3_truth) = run_exp(ground_truth, S1_list, max_diff, num_data_points)

        (diff1, diff2, diff3,
         output1_approx, output2_approx, output3_approx) = run_exp(approximation, S1_list, max_diff, num_data_points)

        # collect plot data
        x1_data = diff1
        x2_data = diff2
        x3_data = diff3
        y1_truth_data = output1_truth
        y2_truth_data = output2_truth
        y3_truth_data = output3_truth
        y1_approx_data = output1_approx
        y2_approx_data = output2_approx
        y3_approx_data = output3_approx

        x_limit = [0, max_diff]
        x_label = "$\Delta I$"
        y_label = "$P($Detected$)$"
        legend = list()
        for S1 in S1_list:
            legend.append('$I_0$ = ' + str(S1)+ ' (Orig.)')
        for S1 in S1_list:
            legend.append('$I_0$ = ' + str(S1)+ ' (Recov.)')

        # plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import pyplot
        import os

        fig, ax = pyplot.subplots(figsize=figure_dimensions)

        ax.plot(x1_data, y1_truth_data, '-', label=legend[0], color='#CC6677')
        ax.plot(x2_data, y2_truth_data, '-', label=legend[1], color='#44AA99')
        ax.plot(x3_data, y3_truth_data, '-', label=legend[2], color='#332288')
        ax.plot(x1_data, y1_approx_data, '--', label=legend[3], color='#CC6677')
        ax.plot(x2_data, y2_approx_data, '--', label=legend[4], color='#44AA99')
        ax.plot(x3_data, y3_approx_data, '--', label=legend[5], color='#332288')
        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        ax.set_xlabel(x_label, fontsize=axis_font_size)
        ax.set_ylabel(y_label, fontsize=axis_font_size)
        ax.set_title('Psychometric Function', fontsize=title_font_size)
        ax.legend(loc=0, fontsize=legend_font_size, bbox_to_anchor=(1.05, 1))
        sns.despine(trim=True)
        plt.show()

        if figure_name is not None and figures_path is not None:
            if not os.path.exists(figures_path):
                os.mkdir(figures_path)
            fig.savefig(os.path.join(figures_path, figure_name))


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
        input = torch.empty(1, 2)
        input[0, 0] = S1
        input[0, 1] = S2
        output = torch.sigmoid(model(input)).detach().numpy().flatten()[0]
        diff1.append(S2-S1)
        output1.append(output)

    S1 = S1_list[1]
    S2_list = np.linspace(S1, S1 + max_diff, num_data_points)
    for S2 in S2_list:
        input[0, 0] = S1
        input[0, 1] = S2
        output = torch.sigmoid(model(input)).detach().numpy().flatten()[0]
        diff2.append(S2 - S1)
        output2.append(output)

    S1 = S1_list[2]
    S2_list = np.linspace(S1, S1 + max_diff, num_data_points)
    for S2 in S2_list:
        input[0, 0] = S1
        input[0, 1] = S2
        output = torch.sigmoid(model(input)).detach().numpy().flatten()[0]
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
    x_label = "$\Delta I = I_1 - I_0$"
    y_label = "P(Detected)"
    legend = list()
    for S1 in S1_list:
        legend.append('$I_0 = $' + str(S1))

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

model = Weber_Model()
# plot_psychophysics(model)
# participant = Participant_Weber()
# participant.graph_simple('test.pdf')