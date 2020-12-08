import AER_config as AER_cfg
import numpy as np
import AER_experimentalist.experimentalist_config as exp_cfg
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

from AER_experimentalist.experimentalist import Experimentalist


from abc import ABC, abstractmethod

class Experimentalist_Popper(Experimentalist, ABC):

    num_training_epochs = 50000
    num_optimization_epochs = 100
    lr = 0.05
    optim_lr = 0.01
    mse_scale = 100000
    momentum = 0.0
    weight_decay = 0

    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None, seed_data_file="", n_hidden=3):
        super(Experimentalist_Popper, self).__init__(study_name, experiment_server_host, experiment_server_port, seed_data_file)

        self.n_hidden = 3

    def sample_experiment(self, model, object_of_study):
        super(Experimentalist_Popper, self).sample_experiment(model, object_of_study)

        # set up a popper network
        n_input = object_of_study.__get_input_dim__()
        n_hidden = self.n_hidden
        n_output = object_of_study.__get_output_dim__()
        popper_net = PopperNet(n_input, n_hidden, n_output)
        popper_optimizer = optim.SGD(popper_net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # get input patterns
        (input_data, output_data) = object_of_study.get_dataset()
        popper_input = Variable(input_data, requires_grad = False)

        # get target patterns
        model_prediction = model(input_data)
        criterion = nn.MSELoss()
        model_loss = (model_prediction - output_data) ** 2 * self.mse_scale
        popper_target = Variable(model_loss, requires_grad = False)

        # train the popper network to predict the loss of the current model
        loss_log = list()
        for training_epoch in range(self.num_training_epochs):
            # zero out gradients
            popper_optimizer.zero_grad()
            popper_prediction = popper_net(popper_input)
            popper_loss = criterion(popper_prediction, popper_target)
            loss_log.append(popper_loss)
            # compute gradients for model
            popper_loss.backward()
            # applies weight decay to classifier weights
            popper_optimizer.step()

        import matplotlib.pyplot as plt
        plt.plot(loss_log)
        plt.ylabel('some numbers')
        plt.show()
        popper_prediction = popper_net(popper_input)
        image = np.concatenate()

        # todo: store data to generate a few plots
        # learning curve of popper experimentalist
        # pattern comparison of popper experimentalist: np.concatenate((popper_prediction.detach().numpy(), popper_target.detach().numpy()), axis=1)
        #

        # todo: loop through each input pattern
        # once the popper network is trained, invert the network to determine optimal experiment conditions
        for optimization_epoch in range(self.num_optimization_epochs):
            # set up input to popper model
            popper_input_optim = popper_input.clone().detach()
            popper_input_optim.requires_grad = True
            # feedforward pass on popper network
            popper_prediction = popper_net(popper_input_optim)
            # compute gradient that maximizes output of popper network (i.e. predicted loss of original model)
            popper_loss_optim = -popper_prediction
            popper_loss_optim.backwards()
            # apply gradient to input
            popper_input_optim += -self.optim_lr * popper_input_optim.grad

        # todo: turn reuslting input into new experiment conditions


        # return experiment_file_path
        return ""
        # pass


class PopperNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()

        # input to hidden projections
        self.hidden = nn.Linear(n_input, n_hidden, bias=True)
        # hidden to output projections
        self.output = nn.Linear(n_hidden, n_output, bias=True)

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)

        return x
