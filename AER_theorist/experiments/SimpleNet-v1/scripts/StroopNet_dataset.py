import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable

try:
    from cnnsimple.StroopNet import StroopNet
    from cnnsimple.object_of_study import objectOfStudy, outputTypes
except:
    from StroopNet import StroopNet
    from object_of_study import objectOfStudy, outputTypes

from torchvision import transforms

class StroopNetDataset(objectOfStudy, Dataset):
    """Stroop model data set."""

    inputDimensions = 6
    outputDimensions = 2

    def __init__(self, num_patterns = 100, sampling = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(StroopNetDataset, self).__init__(num_patterns, sampling)

        # self._num_patterns = num_patterns
        self.transform = transforms.Compose([transforms.ToTensor()])

        # generate input stimuli
        self.color_green = torch.zeros(num_patterns, 1)
        self.color_red = torch.zeros(num_patterns, 1)
        self.word_red = torch.zeros(num_patterns, 1)
        self.word_green = torch.zeros(num_patterns, 1)
        self.task_color = torch.zeros(num_patterns, 1)
        self.task_word = torch.zeros(num_patterns, 1)

        for idx in range(num_patterns):

            # flip a coin to determine which color to present
            sample_color = torch.rand(1)
            if sample_color.numpy() > 0.5:
                self.color_green[idx,:] = sample_color
            else:
                self.color_red[idx, :] = sample_color

            # flip a coin to determine which word to present
            sample_word = torch.rand(1)
            if sample_word.numpy() > 0.5:
                self.word_green[idx, :] = sample_word
            else:
                self.word_red[idx, :] = sample_word

            # flip a coin to determine which task to present
            sample_task = torch.rand(1)
            if sample_task.numpy() > 0.5:
                self.task_color[idx, :] = sample_task
            else:
                self.task_word[idx, :] = sample_task


        self.finalResponse = generateLabels(
            self.color_green, self.color_red,
            self.word_green, self.word_red,
            self.task_color, self.task_word,
            self._sampling)


    def __len__(self):
        return self._num_patterns

    def __getitem__(self, idx):

        input = torch.cat([self.color_green[idx],
                           self.color_red[idx],
                           self.word_green[idx],
                           self.word_red[idx],
                           self.task_color[idx],
                           self.task_word[idx]])
        category = self.finalResponse[idx].data

        return input, category

    def __get_input_dim__(self):
        return StroopNetDataset.inputDimensions

    def __get_output_dim__(self):
        return StroopNetDataset.outputDimensions

    def __get_output_type__(self):
        if self._sampling is True:
            return outputTypes.CLASS
        else:
            return outputTypes.PROBABILITY_DISTRIBUTION

    def __get_input_labels__(self):
        input_labels = list()
        input_labels.append('COLOR_red')
        input_labels.append('COLOR_green')
        input_labels.append('WORD_red')
        input_labels.append('WORD_green')
        input_labels.append('TASK_color')
        input_labels.append('TASK_word')
        return input_labels

    def __get_name__(self):
        return 'StroopNet'

    def sample_model_fit(self, num_patterns, estimator):

        tmp_sampling = self._sampling

        # vary strength of color unit for incongruent Stroop stimulus (demand effect)

        color_green = torch.ones(num_patterns, 1)
        color_red = torch.zeros(num_patterns, 1)
        word_green = torch.zeros(num_patterns, 1)
        word_red = torch.ones(num_patterns, 1)
        task_color = torch.linspace(0, 1, num_patterns)
        task_color = task_color[:, None]
        task_word = torch.zeroes(num_patterns, 1)

        input = torch.cat((color_green, color_red, word_green, word_red, task_color, task_word),1)

        # generate true labels
        response = self.generateLabels(color_green, color_red, word_green, word_red, task_color, task_word, self._sampling)

        self._sampling = tmp_sampling

        # sample data from estimator
        input_var = Variable(input)
        target_var = response
        softmax = nn.Softmax(dim=1)
        prediction_plot = softmax(estimator(input_var)).data.numpy()[:, 0]

        # data to plot (from true model)
        input_plot = input_var.data.numpy()[:, 4]
        target_plot = response.data.numpy()[:, 0]

        return input_plot, target_plot, prediction_plot, input_var, target_var


    def sample_model_fit_2d(self, num_patterns, estimator):

        tmp_sampling = self._sampling
        self._sampling = False

        stimulus1 = np.outer(np.linspace(0, 1, num_patterns), np.ones(num_patterns))
        stimulus2 = stimulus1.copy().T

        # plot variables
        input1_plot = stimulus1
        input2_plot = stimulus2
        target_plot = np.empty(shape=input1_plot.shape)
        prediction_plot = np.empty(shape=input1_plot.shape)

        # test patterns
        input_var = Variable(torch.from_numpy(np.empty(shape=(num_patterns*num_patterns, 6))).float())
        target_var = Variable(torch.from_numpy(np.empty(shape=(num_patterns*num_patterns, 2))).float())

        for col in range(num_patterns):

            # get stimulus vector from grid
            stim1 = torch.from_numpy(stimulus1[:, col])
            stim2 = torch.from_numpy(stimulus2[:, col])
            stim1 = stim1[:, None].float()
            stim2 = stim2[:, None].float()

            total_length = len(stim2)
            color_green = torch.ones(total_length, 1)
            color_red = torch.zeros(total_length, 1)
            word_green = torch.zeros(total_length, 1)
            word_red = stim1
            task_color = stim2
            task_word = torch.zeros(num_patterns, 1)

            input = torch.cat((color_green, color_red, word_green, word_red, task_color, task_word), 1)

            # generate true labels
            response = generateLabels(
                color_green, color_red,
                word_green, word_red,
                task_color, task_word,
                self._sampling)
            target_plot[:, col] = response[:, 0]

            # sample data from estimator
            input_var_tmp = Variable(input)
            softmax = nn.Softmax(dim=1)
            prediction_plot[:, col] = softmax(estimator(input_var_tmp)).data.numpy()[:, 0]

            # store pattern
            input_var.data[(col * num_patterns):((col + 1) * num_patterns), :] = input
            target_var.data[(col * num_patterns):((col + 1) * num_patterns), :] = response.data

        self._sampling = tmp_sampling

        return input1_plot, input2_plot, target_plot, prediction_plot, input_var, target_var


def generateLabels(color_green, color_red, word_green, word_red, task_color, task_word, sampling=False):

    num_patterns = len(color_green)

    colors = torch.zeros(num_patterns, 2)
    words = torch.zeros(num_patterns, 2)
    tasks = torch.zeros(num_patterns, 2)

    colors[:, 0] = color_green
    colors[:, 1] = color_red

    words[:, 0] = word_green
    words[:, 1] = word_red

    tasks[:, 0] = task_color
    tasks[:, 1] = task_word

    # generate labels
    model = StroopNet()  # create instance of StroopNet
    out = model(Variable(colors), Variable(words), Variable(tasks))  # get soft-maxed response pattern
    if sampling:
        uniformSample = Variable(torch.rand(num_patterns))  # get uniform sample for each response
        finalResponse = torch.zeros(num_patterns)
        for i, (response, sample) in enumerate(zip(out, uniformSample)):  # determine final response
            finalResponse[i] = 0 if (response[0] > sample).all() == 1 else 1
        finalResponse = Variable(finalResponse.long())
    else:
        finalResponse = out

    return finalResponse

def get_target(stimulus_pattern, sampled=False):

    color_green = stimulus_pattern.data[:, 0]
    color_red = stimulus_pattern.data[:, 1]
    word_green = stimulus_pattern.data[:, 2]
    word_red = stimulus_pattern.data[:, 3]
    task_color = stimulus_pattern.data[:, 4]
    task_word = stimulus_pattern.data[:, 5]

    if(len(color_green.shape) == 1):
        color_green = color_green.unsqueeze(1)
        color_red = color_red.unsqueeze(1)
        word_green = word_green.unsqueeze(1)
        word_red = word_red.unsqueeze(1)
        task_color = task_color.unsqueeze(1)
        task_word = task_word.unsqueeze(1)
    return generateLabels(
        color_green, color_red,
        word_green, word_red,
        task_color, task_word,
        sampling=sampled)
