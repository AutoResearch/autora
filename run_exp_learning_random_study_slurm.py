import argparse
from datetime import datetime

import aer_experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.object_of_study import ObjectOfStudy
from aer.variable import DVInSilico as DV
from aer.variable import IVInSilico as IV
from aer.variable import ValueType as output_type
from aer_experimentalist.experimentalist_popper import Experimentalist_Popper
from aer_theorist.theorist_random_darts import Theorist_Random_DARTS

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument("--slurm_id", type=int, default=1, help="number of slurm array")
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

# SIMULATION PARAMETERS

study_name = "Exp Learning ICML"  # "Exp Learning 3"   # name of experiment
max_num_data_points = 500

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
learning_trial = IV(
    name="learning_trial", value_range=(0, 1), units="trial", variable_label="Trial"
)

P_initial = IV(
    name="P_initial",
    value_range=(0, 0.4),
    units="accuracy",
    variable_label="Initial Performance",
)

P_asymptotic = IV(
    name="P_asymptotic",
    value_range=(0.5, 1),
    units="accuracy",
    variable_label="Best Performance",
)


# specify dependent variable with type
learning_performance = DV(
    name="learning_performance",
    value_range=(0, 1),
    units="probability",
    variable_label="Accuracy",
    type=output_type.REAL,
)  # not a probability because sum of activations may exceed 1


# list dependent and independent variables
IVs = [
    learning_trial,
    P_initial,
    P_asymptotic,
]  # only including subset of available variables
DVs = [learning_performance]

# initialize objects of study
study_object = ObjectOfStudy(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimentalist_Popper(
    study_name=study_name,
    experiment_server_host=host,
    experiment_server_port=port,
)
# THEORIST

# initialize theorist
theorist = Theorist_Random_DARTS(study_name, theorist_filter="darts")

# specify plots
plots = list()
plots.append(theorist._loss_plot_name)
# for i in range(20):
#     plot_name = "Edge " + str(i)
#     plots.append(plot_name)
# theorist.plot(plot=True, plot_name_list=plots)

# AUTONOMOUS EMPIRICAL RESEARCH

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object, n=max_num_data_points) # seed with new experiment
seed_data = experimentalist.seed(
    study_object, datafile="experiment_0_data.csv"
)  # seed with existing data file
study_object.add_data(seed_data)

# add validation set
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "validation loss"
theorist.add_validation_set(validation_object_2, "validation loss")

# search model ORIGINAL
model = theorist.search_model_job(study_object, args.slurm_id)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
