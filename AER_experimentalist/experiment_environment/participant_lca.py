from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import torch
import torch.nn as nn
from graphviz import Digraph
import random
import numpy as np

class LCA_Model(nn.Module):
    def __init__(self, input=0, gamma=0.4, alpha=0.2, beta=0.2, dt_tau=1):
        super(LCA_Model, self).__init__()

        self.input = input
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.dt_tau = dt_tau

    def forward(self, input):

        input = torch.Tensor(input)
        if len(input.shape) <= 1:
            input = input.view(1, len(input))

        # convert inputs
        x1 = torch.zeros(input.shape[0], 1)
        x2 = torch.zeros(input.shape[0], 1)
        x3 = torch.zeros(input.shape[0], 1)

        x1[:, 0] = input[:, 0]
        x2[:, 0] = input[:, 1]
        x3[:, 0] = input[:, 2]

        d_x1 = (self.input - self.gamma * x1 + self.alpha * torch.relu(x1) - self.beta * (torch.relu(x2) + torch.relu(x3))) * self.dt_tau

        output = d_x1

        return output

class Participant_LCA(Participant_In_Silico):

    # initializes participant
    def __init__(self, input=0, gamma=0.4, alpha=0.2, beta=0.2, dt_tau=1):
        super(Participant_LCA, self).__init__()

        self.x1_lca = torch.ones(1, 1)
        self.x2_lca = torch.ones(1, 1)
        self.x3_lca = torch.zeros(1, 1)

        self.output = torch.zeros(1, 1)
        self.output_sample = 0
        self.model = LCA_Model(input=input, gamma=gamma, alpha=alpha, beta=beta, dt_tau=dt_tau)

    # read value from participant
    def get_value(self, variable_name):

        if variable_name is "x1_lca":
            return self.x1_lca[0, 0].numpy()

        elif variable_name is "x2_lca":
            return self.x2_lca[0, 0].numpy()

        elif variable_name is "x3_lca":
            return self.x3_lca[0, 0].numpy()

        elif variable_name is "dx1_lca":
            return self.output[0, 0].numpy()

        raise Exception('Could not get value from LCA Participant. Variable name "' + variable_name + '" not found.')

    # assign value to participant
    def set_value(self, variable_name, value):

        if variable_name is "x1_lca":
            self.x1_lca[0, 0] = value

        elif variable_name is "x2_lca":
            self.x2_lca[0, 0] = value

        elif variable_name is "x3_lca":
            self.x3_lca[0, 0] = value

        elif variable_name is "dx1_lca":
            self.output[0, 0] = value

        else:
            raise Exception('Could not set value for LCA Participant. Variable name "' + variable_name + '" not found.')

    def execute(self):

        input = torch.zeros(1, 3)
        input[0, 0] = self.x1_lca
        input[0, 1] = self.x2_lca
        input[0, 2] = self.x3_lca

        # compute regular output
        self.output = self.model(input).detach()

    def compute_BIC(self, object_of_study, num_params = None):
        raise Exception('Not implemented.')

    def graph_simple(self, filepath):
        raise Exception("Not implemented.")

    def figure_plot(self, comparison_model,
                    x1=0.5,
                    x2=0.1,
                    x3=0.2,
                    n_trials=20,
                    figures_path=None,
                    figure_name=None,
                    figure_dimensions=(4, 3),
                    y_limit=None,
                    legend_font_size=8,
                    axis_font_size=10,
                    title_font_size=10):

        ground_truth = self.model
        approximation = comparison_model

        (x1_truth_log, x2_truth_log, x3_truth_log) = run_exp(ground_truth, x1=x1, x2=x2, x3=x3, n_trials=n_trials)
        (x1_approx_log, x2_approx_log, x3_approx_log) = run_exp(approximation, x1=x1, x2=x2, x3=x3, n_trials=n_trials)

        # collect plot data
        x_data = range(n_trials)
        y1_truth_data = x1_truth_log
        y2_truth_data = x2_truth_log
        y3_truth_data = x3_truth_log
        y1_approx_data = x1_approx_log
        y2_approx_data = x2_approx_log
        y3_approx_data = x3_approx_log

        x_limit = [0, n_trials]
        x_label = "Time Step"
        y_label = "Net Input $x_i$"
        legend = list()
        legend.append('$x_1$ (Orig.)') # Ground Truth
        legend.append('$x_2$ (Orig.)')
        legend.append('$x_3$ (Orig.)')
        legend.append('$x_1$ (Recov.)')
        legend.append('$x_2$ (Recov.)')
        legend.append('$x_3$ (Recov.)')

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
        if y_limit is not None:
            ax.set_ylim(y_limit)
        ax.set_xlabel(x_label, fontsize=axis_font_size)
        ax.set_ylabel(y_label, fontsize=axis_font_size)
        ax.set_title('LCA Dynamics', fontsize=title_font_size)
        # ax.legend(loc=1, fontsize=legend_font_size)
        ax.legend(loc=0, bbox_to_anchor=(1.05, 1), title='', fontsize=legend_font_size)
        sns.despine(trim=True)
        plt.show()

        if figure_name is not None and figures_path is not None:
            if not os.path.exists(figures_path):
                os.mkdir(figures_path)
            fig.savefig(os.path.join(figures_path, figure_name))

def run_exp(model, x1, x2, x3, n_trials=20):
    x1_log = list()
    x2_log = list()
    x3_log = list()

    for trial in range(n_trials):

        x1_log.append(x1)
        x2_log.append(x2)
        x3_log.append(x3)

        # compute dx1
        input = torch.empty(1, 3)
        input[0, 0] = x1
        input[0, 1] = x2
        input[0, 2] = x3
        dx1 = model(input).detach().numpy().flatten()[0]

        # compute dx2
        input = torch.empty(1, 3)
        input[0, 0] = x2
        input[0, 1] = x1
        input[0, 2] = x3
        dx2 = model(input).detach().numpy().flatten()[0]

        # compute dx3
        input = torch.empty(1, 3)
        input[0, 0] = x3
        input[0, 1] = x1
        input[0, 2] = x2
        dx3 = model(input).detach().numpy().flatten()[0]

        x1 = x1 + dx1
        x2 = x2 + dx2
        x3 = x3 + dx3

    return x1_log, x2_log, x3_log


def plot_trajectory(model, x1=0.5, x2=0.1, x3=0.2, n_trials=20):

    (x1_log, x2_log, x3_log) = run_exp(model, x1=x1, x2=x2, x3=x3, n_trials=n_trials)

    # collect plot data
    x1_data = range(n_trials)
    x2_data = range(n_trials)
    x3_data = range(n_trials)
    y1_data = x1_log
    y2_data = x2_log
    y3_data = x3_log

    x_limit = [0, n_trials]
    y_limit = [-0.5, 0.5]
    x_label = "Time Step"
    y_label = "x"
    legend = list()
    legend.append('x1(0) = ' + str(x1))
    legend.append('x2(0) = ' + str(x2))
    legend.append('x3(0) = ' + str(x3))

    # plot
    import matplotlib.pyplot as plt
    plt.plot(x1_data, y1_data, '.-', label=legend[0])
    plt.plot(x2_data, y2_data, '.-', label=legend[1])
    plt.plot(x3_data, y3_data, '.-', label=legend[2])
    plt.xlim(x_limit)
    # plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=1, fontsize="large")
    plt.show()

# model = LCA_Model()
# plot_trajectory(model)