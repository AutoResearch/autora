import argparse
from datetime import datetime
from typing import List

import aer.experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import DARTSType, Theorist_DARTS
from aer.variable import DVInSilico as DV
from aer.variable import IVInSilico as IV
from aer.variable import ValueType as output_type

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument("--slurm_id", type=int, default=0, help="number of slurm array")
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

# SIMULATION PARAMETERS

study_name = "LCA ICML"  # LCA  # name of experiment
max_num_data_points = 500

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
x1 = IV(name="x1_lca", value_range=(-1, 1), units="net input", variable_label="x1")

x2 = IV(name="x2_lca", value_range=(-1, 1), units="net input", variable_label="x2")

x3 = IV(name="x3_lca", value_range=(-1, 1), units="net input", variable_label="x3")


# specify dependent variable with type
dx1_lca = DV(
    name="dx1_lca",
    value_range=(0, 1),
    units="net input change",
    variable_label="dx1",
    type=output_type.REAL,
)  # not a probability because sum of activations may exceed 1


# list dependent and independent variables
IVs = [x1, x2, x3]  # only including subset of available variables
DVs = [dx1_lca]

# initialize objects of study
study_object = Object_Of_Study(
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
theorist = Theorist_DARTS(study_name, darts_type=DARTSType.ORIGINAL)

# specify plots
plots: List[str] = list()
# plots.append(theorist._loss_plot_name)
# for i in range(20):
#     plot_name = "Edge " + str(i)
#     plots.append(plot_name)
theorist.plot(plot=False, plot_name_list=plots)

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
