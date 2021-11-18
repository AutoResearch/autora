import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable

from AER_theorist.darts.SimpleNet import SimpleNet
from AER_theorist.darts.object_of_study import objectOfStudy, outputTypes

from torchvision import transforms

class SimpleNetDataset(objectOfStudy, Dataset):
    """Stroop model data set."""

    inputDimensions = 2
    outputDimensions = 2

    def __init__(self, num_patterns = 100, sampling = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(SimpleNetDataset, self).__init__(num_patterns, sampling)

        # self._num_patterns = num_patterns
        self.transform = transforms.Compose([transforms.ToTensor()])

        # generate input stimuli
        self.stimulus1 = torch.rand(num_patterns, 1)  # values for first stimulus are drawn from U(0,1)
        self.stimulus2 = torch.rand(num_patterns, 1) * 0 # values for second stimulus are drawn from U(0,1)

        self.finalResponse = generateLabels(self.stimulus1, self.stimulus2, self._sampling)


    def __len__(self):
        return self._num_patterns

    def __getitem__(self, idx):

        input = torch.cat([self.stimulus1[idx], self.stimulus2[idx]])
        category = self.finalResponse[idx].data


        # sample = {'input': input, 'category': category}

        # if self.transform:
        #     sample = self.transform(sample)

        return input, category

    def __get_input_dim__(self):
        return SimpleNetDataset.inputDimensions

    def __get_output_dim__(self):
        return SimpleNetDataset.outputDimensions

    def __get_output_type__(self):
        if self._sampling is True:
            return outputTypes.CLASS
        else:
            return outputTypes.PROBABILITY_DISTRIBUTION

    def __get_input_labels__(self):
        input_labels = list()
        input_labels.append('stim1')
        input_labels.append('stim2')
        return input_labels

    def __get_name__(self):
        return 'SimpleNet'

    def sample_model_fit(self, num_patterns, estimator):

        tmp_sampling = self._sampling

        stimulus2 = torch.zeros(num_patterns, 1) # since the input to hidden layer weight for this stimulus is set to 0, it should have no impact on the model's outcome
        stimulus1 = torch.linspace(0, 1, num_patterns)
        stimulus1 = stimulus1[:, None]

        input = torch.cat((stimulus1, stimulus2),1)

        # generate true labels
        response = self.generateLabels(stimulus1, stimulus2, self._sampling)

        self._sampling = tmp_sampling

        # sample data from estimator
        input_var = Variable(input)
        target_var = response
        softmax = nn.Softmax(dim=1)
        prediction_plot = softmax(estimator(input_var)).data.numpy()[:, 0]

        # data to plot (from true model)
        input_plot = input_var.data.numpy()[:, 0]
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
        input_var = Variable(torch.from_numpy(np.empty(shape=(num_patterns*num_patterns, 2))).float())
        target_var = Variable(torch.from_numpy(np.empty(shape=(num_patterns*num_patterns, 2))).float())

        for col in range(num_patterns):

            # get stimulus vector from grid
            stim1 = torch.from_numpy(stimulus1[:, col])
            stim2 = torch.from_numpy(stimulus2[:, col])
            stim1 = stim1[:, None].float()
            stim2 = stim2[:, None].float()

            input = torch.cat((stim1, stim2), 1)

            # generate true labels
            response = generateLabels(stim1, stim2, self._sampling)
            target_plot[:, col] = response.detach().numpy()[:, 0]

            # sample data from estimator
            input_var_tmp = Variable(input)
            softmax = nn.Softmax(dim=1)
            prediction_plot[:, col] = softmax(estimator(input_var_tmp)).data.numpy()[:, 0]

            # store pattern
            input_var.data[(col * num_patterns):((col + 1) * num_patterns), :] = input
            target_var.data[(col * num_patterns):((col + 1) * num_patterns), :] = response.data

        self._sampling = tmp_sampling

        return input1_plot, input2_plot, target_plot, prediction_plot, input_var, target_var


def generateLabels(stimulus1, stimulus2, sampling=False):

    num_patterns = len(stimulus1)

    # generate labels
    model = SimpleNet()  # create instance of SimpleNet
    out = model(Variable(stimulus1), Variable(stimulus2))  # get soft-maxed response pattern
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
    stimulus1 = stimulus_pattern.data[:, 0]
    stimulus2 = stimulus_pattern.data[:, 1]
    if(len(stimulus1.shape) == 1):
        stimulus1 = stimulus1.unsqueeze(1)
        stimulus2 = stimulus2.unsqueeze(1)
    return generateLabels(stimulus1, stimulus2, sampling=sampled)
