import numpy as np
import torch
import torch.nn as nn
from sweetpea import *
from torch.autograd import Variable

from autora.variable import DV, IV, ValueType, VariableCollection

# Stroop Model
stroop_stimulus_resolution = 10
stroop_choice_temperature = 1.0
added_noise = 0


def stroop_model_metadata():

    color_green = IV(
        name="color_green",
        allowed_values=np.linspace(0, 1, stroop_stimulus_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Green",
        type=ValueType.REAL,
    )

    color_red = IV(
        name="color_red",
        allowed_values=np.linspace(0, 1, stroop_stimulus_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Red",
        type=ValueType.REAL,
    )

    word_green = IV(
        name="word_green",
        allowed_values=np.linspace(0, 1, stroop_stimulus_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Word GREEN",
        type=ValueType.REAL,
    )

    word_red = IV(
        name="word_red",
        allowed_values=np.linspace(0, 1, stroop_stimulus_resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Word RED",
        type=ValueType.REAL,
    )

    task_color = IV(
        name="task_color",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Naming Task",
        type=ValueType.REAL,
    )

    task_word = IV(
        name="task_word",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="intensity",
        variable_label="Word Reading Task",
        type=ValueType.REAL,
    )

    response_green = DV(
        name="performance",
        value_range=(0, 1),
        units="percentage",
        variable_label="P(Green Response)",
        type=ValueType.PROBABILITY,
    )

    metadata = VariableCollection(
        independent_variables=[
            color_green,
            color_red,
            word_green,
            word_red,
            task_color,
            task_word,
        ],
        dependent_variables=[response_green],
    )

    return metadata


def stroop_model_experiment(
    X: np.ndarray, choice_temperature=stroop_choice_temperature, std=added_noise
):
    Y = np.zeros((X.shape[0], 1))

    # Stroop Model according to
    # Cohen, J. D., Dunbar, K. M., McClelland, J. L., & Rohrer, D. (1990). On the control of automatic processes: a parallel distributed processing account of the Stroop effect. Psychological review, 97(3), 332.
    model = Stroop_Model(choice_temperature)

    for idx, x in enumerate(X):
        # compute regular output
        output_net = model(x).detach().numpy() + std
        p_choose_A = output_net[0][0]

        Y[idx] = p_choose_A

    return Y


def stroop_model_data(metadata):

    color_intensity = metadata.independent_variables[0].allowed_values
    word_intensity = metadata.independent_variables[2].allowed_values
    color = [0, 1]
    word = [0, 1]
    task = [0, 1]

    conditions = np.array(
        np.meshgrid(color, color_intensity, word, word_intensity, task)
    ).T.reshape(-1, 5)

    X = np.zeros((conditions.shape[0], 6))

    # translate conditions into full X matrix where each intensity is coded separately
    for idx, condition in enumerate(conditions):
        if condition[0] == 0:
            X[idx, 0] = condition[1]
            X[idx, 1] = 0
        else:
            X[idx, 0] = 0
            X[idx, 1] = condition[1]
        if condition[2] == 0:
            X[idx, 2] = condition[3]
            X[idx, 3] = 0
        else:
            X[idx, 2] = 0
            X[idx, 3] = condition[3]
        if condition[4] == 1:
            X[idx, 4] = 1
            X[idx, 5] = 0
        else:
            X[idx, 4] = 0
            X[idx, 5] = 1

    y = stroop_model_experiment(X, std=0)

    return X, y


class Stroop_Model(nn.Module):
    def __init__(self, choice_temperature):
        super(Stroop_Model, self).__init__()

        self.choice_temperature = choice_temperature

        # define affine transformations
        self.input_color_hidden_color = nn.Linear(2, 2, bias=False)
        self.input_word_hidden_word = nn.Linear(2, 2, bias=False)
        self.hidden_color_output = nn.Linear(2, 2, bias=False)
        self.hidden_word_output = nn.Linear(2, 2, bias=False)
        self.task_hidden_color = nn.Linear(2, 2, bias=False)
        self.task_hidden_word = nn.Linear(2, 2, bias=False)

        self.bias = Variable(torch.ones(1) * -4, requires_grad=False)
        self.input_color_hidden_color.weight.data = (
            torch.FloatTensor([[1, -1], [-1, 1]]) * 2.2
        )
        self.hidden_color_output.weight.data = (
            torch.FloatTensor([[1, -1], [-1, 1]]) * 1.3
        )

        self.input_word_hidden_word.weight.data = (
            torch.FloatTensor([[1, -1], [-1, 1]]) * 2.6
        )
        self.hidden_word_output.weight.data = (
            torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5
        )

        self.task_hidden_color.weight.data = (
            torch.FloatTensor([[1.0, 0.0], [1.0, 0]]) * 4
        )
        self.task_hidden_word.weight.data = torch.FloatTensor([[0, 1], [0, 1]]) * 4

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

        color_hidden = torch.sigmoid(
            self.input_color_hidden_color(color)
            + self.task_hidden_color(task)
            + self.bias
        )

        word_hidden = torch.sigmoid(
            self.input_word_hidden_word(word) + self.task_hidden_word(task) + self.bias
        )

        output = self.hidden_color_output(color_hidden) + self.hidden_word_output(
            word_hidden
        )

        output_softmaxed = torch.exp(output * 1 / self.choice_temperature) / (
            torch.exp(output[:, 0] * 1 / self.choice_temperature)
            + torch.exp(output[:, 1] * 1 / self.choice_temperature)
        )

        return output_softmaxed


def run_exp(model):

    # color naming - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 1
    task_word = 0
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_col_cong = model(input)  # torch.sigmoid(model(input))

    # color naming - incong
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 1
    task_color = 1
    task_word = 0
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_col_incong = model(input)

    # color naming - control
    color_red = 1
    color_green = 0
    word_red = 0
    word_green = 0
    task_color = 1
    task_word = 0
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_col_control = model(input)

    # word reading - cong
    color_red = 1
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_wrd_cong = model(input)

    # word reading - incong
    color_red = 0
    color_green = 1
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_wrd_incong = model(input)

    # word reading - control
    color_red = 0
    color_green = 0
    word_red = 1
    word_green = 0
    task_color = 0
    task_word = 1
    input = [color_green, color_red, word_green, word_red, task_color, task_word]
    output_wrd_control = model(input)

    return (
        output_col_cong,
        output_col_incong,
        output_col_control,
        output_wrd_cong,
        output_wrd_incong,
        output_wrd_control,
    )


def plot_stroop_model(model=None):

    original_model = Stroop_Model(stroop_choice_temperature)

    # collect plot data for orignal model
    (
        output_col_cong,
        output_col_incong,
        output_col_control,
        output_wrd_cong,
        output_wrd_incong,
        output_wrd_control,
    ) = run_exp(original_model)

    err_col_cong = 1 - output_col_cong[0, 1]
    err_col_incong = 1 - output_col_incong[0, 1]
    err_col_control = 1 - output_col_control[0, 1]
    err_wrd_cong = 1 - output_wrd_cong[0, 1]
    err_wrd_incong = 1 - output_wrd_incong[0, 1]
    err_wrd_control = 1 - output_wrd_control[0, 1]

    x_data = [0, 1, 2]
    y_data_col = [
        err_col_control.detach().numpy() * 100,
        err_col_incong.detach().numpy() * 100,
        err_col_cong.detach().numpy() * 100,
    ]
    y_data_wrd = [
        err_wrd_control.detach().numpy() * 100,
        err_wrd_incong.detach().numpy() * 100,
        err_wrd_cong.detach().numpy() * 100,
    ]

    # collect plot data for recovered model
    if model is not None:
        (
            output_col_cong,
            output_col_incong,
            output_col_control,
            output_wrd_cong,
            output_wrd_incong,
            output_wrd_control,
        ) = run_exp(model)

        err_col_cong = 1 - output_col_cong[0, 1]
        err_col_incong = 1 - output_col_incong[0, 1]
        err_col_control = 1 - output_col_control[0, 1]
        err_wrd_cong = 1 - output_wrd_cong[0, 1]
        err_wrd_incong = 1 - output_wrd_incong[0, 1]
        err_wrd_control = 1 - output_wrd_control[0, 1]

        x_data = [0, 1, 2]
        y_data_col_recovered = [
            err_col_control.detach().numpy() * 100,
            err_col_incong.detach().numpy() * 100,
            err_col_cong.detach().numpy() * 100,
        ]
        y_data_wrd_recovered = [
            err_wrd_control.detach().numpy() * 100,
            err_wrd_incong.detach().numpy() * 100,
            err_wrd_cong.detach().numpy() * 100,
        ]

    x_limit = [-0.5, 2.5]
    y_limit = [0, 50]
    y_label = "Error Rate (%)"
    legend = (
        "Color Naming (Original)",
        "Word Reading (Original)",
        "Color Naming (Recovered)",
        "Word Reading (Recovered)",
    )

    # plot
    import matplotlib.pyplot as plt

    plt.plot(x_data, y_data_col, label=legend[0])
    plt.plot(x_data, y_data_wrd, label=legend[1])
    if model is not None:
        plt.plot(x_data, y_data_col_recovered, "--", label=legend[2])
        plt.plot(x_data, y_data_wrd_recovered, "--", label=legend[3])
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=2, fontsize="large")
    plt.title("Stroop Effect", fontsize="large")
    plt.xticks(x_data, ["Neutral", "Incongruent", "Congruent"], rotation="horizontal")
    plt.show()


# meta_data = stroop_model_metadata()
# X, y = stroop_model_data(meta_data)
# plot_stroop_model()
