import aer.experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from aer.experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from aer.experimentalist.experiment_environment.participant_weber import (
    Participant_Weber,
)
from aer.experimentalist.experiment_environment.variable import (
    outputTypes as output_type,
)
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.darts.plot_utils import plot_darts_summary
from aer.theorist.object_of_study import Object_Of_Study

participant = Participant_Weber()

study_name = "Weber"  # name of experiment
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
S1 = IV(
    name="S1",
    value_range=(0, 1),
    units="intensity",
    variable_label="Stimulus 1 Intensity",
)

S2 = IV(
    name="S2",
    value_range=(0, 1),
    units="intensity",
    variable_label="Stimulus 2 Intensity",
)


# specify dependent variable with type

diff_detected_sample = DV(
    name="difference_detected_sample",
    value_range=(0, 1),
    units="response",
    variable_label="difference detected",
    type=output_type.PROBABILITY_SAMPLE,
)

IVs = [S1, S2]  # only including subset of available variables
DVs_validation = [diff_detected_sample]

validation_object_1 = Object_Of_Study(
    name="Weber Sampled", independent_variables=IVs, dependent_variables=DVs_validation
)

experimentalist_validation = Experimentalist_Popper(
    study_name="Weber Sampled",
    experiment_server_host=host,
    experiment_server_port=port,
)

validation_data = experimentalist_validation.seed(
    validation_object_1, datafile="experiment_0_data.csv"
)  # seed with new experiment
validation_object_1.add_data(validation_data)

# BIC = participant.compute_BIC(object_of_study=validation_object_1, num_params=10)
# y_reference=BIC
# y_reference_label='Data Generating Model'

study_name = "Weber ICML"
# y_name = 'loss'
# y_label = 'log loss'
# y_name = 'Weber_Original'
# y_label = 'log loss (validation)'
# y_name = 'Weber_Sampled'
# y_label = 'BIC'
# y_name = 'num_params'
# y_label = 'Number of Parameters'
y_name = "num_edges"
y_label = "Number of Edges"

# x1_name = 'num_params'
x1_label = r"Parameter Complexity $\gamma$"
x1_name = "arch_weight_decay"
# x1_name = 'num_edges'
# x1_label = 'num edges'

x2_name = "num_graph_node"
x2_label = "k"
# x2_name = 'num_params'
# x2_label = 'num params'
# x2_label = 'df'
# x2_name = 'arch_weight_decay'
x_limit = [-0.005, 0.105]
y_limit = [0, 15]

# plot_darts_summary(study_name=study_name,
#                    y_name=y_name,
#                    x1_name=x1_name,
#                    y_label=y_label,
#                    x1_label=x1_label,
#                    metric='min',
#                    theorist_filter='fair_darts')


plot_darts_summary(
    study_name=study_name,
    y_name=y_name,
    x1_name=x1_name,
    x2_name=x2_name,
    y_label=y_label,
    x1_label=x1_label,
    x2_label=x2_label,
    metric="mean",
    x_limit=x_limit,
    y_limit=y_limit,
    theorist_filter="fair_darts",
)
