from abc import ABC, abstractmethod
import torch
import numpy as np

class Participant_In_Silico:

    def __init__(self):
        pass

    # read value from participant
    @abstractmethod
    def get_value(self, variable_label):
        pass

    # assign value to participant
    @abstractmethod
    def set_value(self, variable_label, value):
        pass

    def compute_BIC(self, input, target, output_function, num_params = None):

        # compute raw model output
        classifier_output = self.model(input)

        # compute associated probability
        prediction = output_function(classifier_output).detach()

        target_flattened = torch.flatten(target.long())
        llik = 0
        n = len(target_flattened)  # number of data points

        # in case there is only a single output (probability)
        if classifier_output.shape[1] == 1:
            llik = 0
            for idx in range(len(target_flattened)):

                # fail safe if model doesn't produce probabilities
                if prediction[idx] > 1:
                    prediction[idx] = 1
                elif prediction[idx] < 0:
                    prediction[idx] = 0

                if target_flattened[idx] == 1:
                    lik = prediction[idx]
                elif target_flattened[idx] == 0:
                    lik = 1 - prediction[idx]
                else:
                    raise Exception('Target must contain either zeros or ones.')
                llik += np.log(lik)

        else:
            for idx in range(len(target_flattened)):
                lik = prediction[idx, target_flattened[idx]]
                llik += np.log(lik)

        if num_params is None:
            k = self.count_parameters()  # for most likely architecture
        else:
            k = num_params
        BIC = np.log(n) * k - 2 * llik

        return BIC.numpy()

    def count_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())

    def figure_plot(self, comparison_model):
        raise Exception('Not implemented.')
