from datetime import datetime

import aer.experimentalist.experiment_environment.experiment_config as exp_cfg
import aer.theorist.darts.utils as utils
import aer.theorist.darts.visualize as viz
from aer.experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from aer.variable import ValueType, Variable

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

study_name = "Weber"  # name of experiment
study_name_sampled = "Weber Sampled"
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
S1 = Variable(
    name="S1",
    value_range=(0, 5),
    units="intensity",
    variable_label="Stimulus 1 Intensity",
)

S2 = Variable(
    name="S2",
    value_range=(0, 5),
    units="intensity",
    variable_label="Stimulus 2 Intensity",
)


# specify dependent variable with type
diff_detected = Variable(
    name="difference_detected",
    value_range=(0, 1),
    units="probability",
    variable_label="P(difference detected)",
    type=ValueType.SIGMOID,
)

diff_detected_sample = Variable(
    name="difference_detected_sample",
    value_range=(0, 1),
    units="response",
    variable_label="difference detected",
    type=ValueType.PROBABILITY_SAMPLE,
)

# list dependent and independent variables
IVs = [S1, S2]  # only including subset of available variables
DVs = [diff_detected]
DVs_validation = [diff_detected_sample]

study_object = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)
# initialize objects of study

validation_object_1 = Object_Of_Study(
    name=study_name_sampled,
    independent_variables=IVs,
    dependent_variables=DVs_validation,
)

# EXPERIMENTALIST

# experiment design
stimulus_resolution = 20
weber_design = Experiment_Design_Synthetic_Weber(stimulus_resolution)

stimulus_resolution_validation = 100
weber_design_validation = Experiment_Design_Synthetic_Weber(
    stimulus_resolution_validation
)

# initialize experimentalist
experimentalist = Experimentalist_Popper(
    study_name=study_name,
    experiment_server_host=host,
    experiment_server_port=port,
    experiment_design=weber_design,
)

experimentalist_validation = Experimentalist_Popper(
    study_name=study_name_sampled,
    experiment_server_host=host,
    experiment_server_port=port,
    experiment_design=weber_design_validation,
)

# THEORIST

# initialize theorist
theorist = Theorist_DARTS(study_name, darts_type=DARTS_Type.ORIGINAL)

# specify plots
# plots = list()
# plots.append(theorist._loss_plot_name)
# theorist.plot()


# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1) # seed with new experiment
validation_data = experimentalist_validation.seed(
    validation_object_1, datafile="experiment_0_data.csv"
)  # seed with new experiment
validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object) # seed with new experiment
seed_data = experimentalist.seed(
    study_object, datafile="experiment_0_data.csv"
)  # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "Weber Sampled"

# add validation sets
theorist.add_validation_set(validation_object_1, "Weber_Sampled")
theorist.add_validation_set(validation_object_2, "Weber_Original")

# search model
model = theorist.search_model(study_object)

# PLOT

# collect relevant variables
genotype = (
    model.genotype().normal
)  # contains information about the architecture of the model
model_graph_filepath = "aer/example/weber/graph.pdf"  # path to store pdf of plot
input_labels = study_object.__get_input_labels__()  # labels of input variables
(
    n_params_total,
    n_params_base,
    param_list,  # list of the parameters for each function
) = model.countParameters()
out_dim = (
    study_object.__get_output_dim__()
)  # dimensionality of the output (# dependent variables)
out_fnc = utils.get_output_str(
    study_object.__get_output_type__()
)  # label specifying the output function

# call to plot function
viz.plot(
    genotype,
    model_graph_filepath,
    viewFile=True,
    input_labels=input_labels,
    param_list=param_list,
    full_label=True,
    out_dim=out_dim,
    out_fnc=out_fnc,
)


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
