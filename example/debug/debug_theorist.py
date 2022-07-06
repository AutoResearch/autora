import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from aer.experimentalist.experiment_environment.variable import Variable as Var
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import Theorist_DARTS


class copyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input to hidden projections
        self.linear = nn.Linear(1, 1, bias=False)
        self.cell = torch.exp
        # hidden to output projections
        self.classifier = nn.Linear(1, 1, bias=True)

    def init_weights(self):
        # self.linear.weight.data.fill_(-0.4654)
        # # self.linear.bias.data.fill_(0.01)
        # self.classifier.weight.data.fill_(-0.3706)
        # self.classifier.bias.data.fill_(0.3601)

        # self.linear.weight.data.fill_(6)
        # self.classifier.weight.data.fill_(0.01)
        # self.classifier.bias.data.fill_(-0.025)

        self.linear.weight.data.fill_(8)
        self.classifier.weight.data.fill_(0.01)
        self.classifier.bias.data.fill_(-0.01)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.linear(x)
        x = self.cell(x)
        x = self.classifier(x)

        return x


# GENERAL PARAMETERS

study_name = "Simple Voltage"  # name of experiment
host = "192.168.188.27"  # exp_env_cfg.HOST_IP      # ip address of experiment server
port = 47778  # exp_env_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variable
source_voltage = Var(
    name="source_voltage",
    value_range=(0, 4000),
    units="mV",
    rescale=0.0001,  # need to convert to V to keep input values small
    variable_label="Source Voltage",
)

# specify dependent variable
target_voltage = Var(
    name="voltage0",
    units="mV",
    rescale=0.0001,  # need to convert to V to keep input values small
    variable_label="Target Voltage",
)

# list dependent and independent variables
IVs = [source_voltage]
DVs = [target_voltage]

# initialize object of study
study_object = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimentalist_Popper(
    study_name=study_name,
    experiment_server_host=host,
    experiment_server_port=port,
    seed_data_file="experiment_0_data.csv",
)

# THEORIST
theorist = Theorist_DARTS(study_name)

# AUTONOMOUS EMPIRICAL RESEARCH

seed_data = experimentalist.seed(study_object, datafile="experiment_0_data.csv")
study_object.add_data(seed_data)

# root = Tk()
# app = Theorist_GUI(object_of_study=study_object, theorist=theorist, root=root)
# root.mainloop()

# generate computational model to explain object of study
model = theorist.search_model(study_object)

# get input
(input, output) = study_object.get_dataset()
prediction = model(input).detach().numpy()

copy_model = copyNet()
copy_model.init_weights()
copy_prediction = copy_model(input).detach().numpy()

# train copy model
lr = 0.005
epochs = 50000
copy_optimizer = optim.SGD(copy_model.parameters(), lr=lr, momentum=0.9)
criterion = nn.MSELoss()
loss_log = list()
for training_epoch in range(epochs):
    copy_optimizer.zero_grad()
    copy_prediction = copy_model(input)
    copy_loss = criterion(copy_prediction, output)
    copy_loss.backward()
    copy_optimizer.step()
    loss_log.append(copy_loss.detach().numpy())

# plot model against data
IV1 = study_object.independent_variables[0]
DV = study_object.dependent_variables[0]
resolution = 100
input = input.numpy()
output = output.numpy()
counterbalanced_input = study_object.get_counterbalanced_input(resolution)
x_prediction = study_object.get_IVs_from_input(counterbalanced_input, IV1).numpy()
y_prediction = model(counterbalanced_input).detach().numpy()
x_data = study_object.get_IVs_from_input(input, IV1)
y_data = study_object.get_DV_from_output(output, DV)
y_limit = [
    np.amin([np.amin(y_data), np.amin(y_prediction)]),
    np.amax([np.amax(y_data), np.amax(y_prediction)]),
]
y_label = DV.get_variable_label()
x_limit = study_object.get_variable_limits(IV1)
x_label = IV1.get_variable_label()

# copy_model.linear.weight.data.fill_(6)
# copy_model.classifier.weight.data.fill_(0.01)
# copy_model.classifier.bias.data.fill_(-0.025)
(input_org, output_org) = study_object.get_dataset()
copy_prediction = copy_model(input_org).detach().numpy()

plt.clf()
plt.plot(x_prediction, y_prediction, "k", label="AER model prediction")
plt.scatter(input, prediction, marker=".", c="k")
plt.scatter(input, copy_prediction, marker=".", c="b")
plt.scatter(input, output, marker=".", c="r", label="data")
plt.ylabel(y_label)
plt.xlabel(x_label)
plt.legend(loc=2, fontsize="small")
plt.show()


plt.clf()
plt.plot(loss_log)
plt.show()
