import AER_config as AER_cfg
import numpy as np
import AER_experimentalist.experimentalist_config as exp_cfg
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import AER_experimentalist.experimentalist_popper_config as popper_config
from AER_utils import Plot_Types

from AER_experimentalist.experimentalist import Experimentalist


from abc import ABC, abstractmethod

class Experimentalist_Popper(Experimentalist, ABC):

    n_hidden = popper_config.n_hidden
    num_training_epochs = popper_config.num_training_epochs
    num_optimization_epochs = popper_config.num_optimization_epochs
    lr = popper_config.lr
    optim_lr = popper_config.optim_lr
    mse_scale = popper_config.mse_scale
    momentum = popper_config.momentum
    weight_decay = popper_config.weight_decay

    _popper_loss_plot_name = "Loss of Popper Network"
    _popper_pattern_plot_name = "Predicted vs. Actual Model Loss"
    popper_loss_log = list()
    popper_pattern_data = None

    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None, seed_data_file=""):
        super(Experimentalist_Popper, self).__init__(study_name, experiment_server_host, experiment_server_port, seed_data_file)


    def init_experiment_search(self, model, object_of_study):
        super(Experimentalist_Popper, self).init_experiment_search(model, object_of_study)

        # set up a popper network
        n_input = object_of_study.__get_input_dim__()
        n_hidden = self.n_hidden
        n_output = object_of_study.__get_output_dim__()
        self.popper_net = PopperNet(n_input, n_hidden, n_output)
        popper_optimizer = optim.SGD(self.popper_net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # get input patterns
        (input_data, output_data) = object_of_study.get_dataset()
        popper_input = Variable(input_data, requires_grad = False)

        # get target patterns
        model_prediction = model(input_data)
        criterion = nn.MSELoss()
        model_loss = (model_prediction - output_data) ** 2 * self.mse_scale
        popper_target = Variable(model_loss, requires_grad = False)

        # train the popper network to predict the loss of the current model
        self.popper_loss_log = list()
        for training_epoch in range(self.num_training_epochs):
            # zero out gradients
            popper_optimizer.zero_grad()
            popper_prediction = self.popper_net(popper_input)
            popper_loss = criterion(popper_prediction, popper_target)
            self.popper_loss_log.append(popper_loss.detach().numpy())
            # compute gradients for model
            popper_loss.backward()
            # applies weight decay to classifier weights
            popper_optimizer.step()

        self.popper_pattern_data = np.concatenate((popper_prediction.detach().numpy(), popper_target.detach().numpy()),
                                                  axis=1)

        # import matplotlib.pyplot as plt
        # plt.plot(self.popper_loss_log)
        # plt.ylabel('some numbers')
        # plt.show()
        return

    def sample_experiment_condition(self, model, object_of_study, condition):

        # todo: loop through each input pattern
        # # once the popper network is trained, invert the network to determine optimal experiment conditions
        # for optimization_epoch in range(self.num_optimization_epochs):
        #     # set up input to popper model
        #     popper_input_optim = popper_input.clone().detach()
        #     popper_input_optim.requires_grad = True
        #     # feedforward pass on popper network
        #     popper_prediction = popper_net(popper_input_optim)
        #     # compute gradient that maximizes output of popper network (i.e. predicted loss of original model)
        #     popper_loss_optim = -popper_prediction
        #     popper_loss_optim.backwards()
        #     # apply gradient to input
        #     popper_input_optim += -self.optim_lr * popper_input_optim.grad

        # todo: turn resulting input into new experiment conditions
        pass

    def get_plots(self, model, object_of_study):
        self.update_popper_learning_curve()
        self.update_popper_pattern_plot()
        return super(Experimentalist_Popper, self).get_plots(object_of_study=object_of_study, model=model)

    def update_popper_learning_curve(self):

        if not self.popper_loss_log:
            return

        # type
        type = Plot_Types.LINE

        # x data
        x = np.linspace(1, len(self.popper_loss_log), len(self.popper_loss_log))

        # y data
        y = np.array(self.popper_loss_log)

        # axis limits
        x_limit = [1, self.num_training_epochs]

        popper_loss_log_np = np.array(self.popper_loss_log)

        if np.isnan(popper_loss_log_np[:]).all():
            y_limit = [0, 1]
        else:
            y_max = np.nanmax(popper_loss_log_np[:])
            y_limit = [0, y_max]

        # axis labels
        x_label = "Epochs"
        y_label = "Loss"

        # legend
        legend = 'Loss'

        # generate plot dictionary
        plot_dict = self._generate_plot_dict(type, x, y, x_limit, y_limit, x_label, y_label, legend)
        self._plots[self._popper_loss_plot_name] = plot_dict


    def update_popper_pattern_plot(self):

        if self.popper_pattern_data is None:
            return

        # type
        type = Plot_Types.IMAGE

        im = self.popper_pattern_data

        # separator
        x = np.ones(int(self.popper_pattern_data.shape[0])) * (self.popper_pattern_data.shape[1]/2 - 0.5)
        y = np.linspace(1, int(self.popper_pattern_data.shape[0]), int(self.popper_pattern_data.shape[0]))

        # axis labels
        x_label = "Predicted (Predicted vs. Actual)"
        y_label = "Experiment Condition"

        # generate plot dictionary
        plot_dict = self._generate_plot_dict(type, x, y, x_label=x_label, y_label=y_label, image=im)
        self._plots[self._popper_pattern_plot_name] = plot_dict


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
