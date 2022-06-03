import aer_config as aer_cfg
import aer_experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.variable import DVInSilico as DV
from aer.variable import IVInSilico as IV
from aer.variable import OutputTypes as output_type
from aer_experimentalist.experiment_environment.participant_weber import (
    Participant_Weber,
)
from aer_experimentalist.experimentalist_popper import Experimentalist_Popper
from aer_theorist.darts.plot_utils import (
    generate_darts_summary_figures,
    load_model,
    plot_model_graph,
)
from aer_theorist.object_of_study import Object_Of_Study

participant = Participant_Weber()

study_name = "Weber ICML"  # name of experiment: "Weber Final"
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

# OBJECT OF STUDY

# specify independent variables
S1 = IV(name="S1", value_range=(0, 1), units="intensity", variable_label="I_0")

S2 = IV(name="S2", value_range=(0, 1), units="intensity", variable_label="I_1")


# specify dependent variable with type

diff_detected = DV(
    name="difference_detected",
    value_range=(0, 1),
    units="response",
    variable_label="difference detected",
    type=output_type.SIGMOID,
)

diff_detected_sample = DV(
    name="difference_detected_sample",
    value_range=(0, 1),
    units="response",
    variable_label="difference detected",
    type=output_type.PROBABILITY_SAMPLE,
)

# list dependent and independent variables
IVs = [S1, S2]
DVs = [diff_detected]
DVs_validation = [diff_detected_sample]

study_object = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)

study_object_validation = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs_validation
)

experimentalist_validation = Experimentalist_Popper(
    study_name="Weber Sampled",
    experiment_server_host=host,
    experiment_server_port=port,
)

validation_data = experimentalist_validation.seed(
    study_object_validation, datafile="experiment_0_data.csv"
)  # seed with new experiment
study_object_validation.add_data(validation_data)

# PLOT MODEL SEARCH RESULTS

figure_names = (
    "legend",
    "model_search_original",
    "model_search_fair",
    "model_search_random",
)
titles = (
    aer_cfg.darts_original_label,
    aer_cfg.darts_original_label,
    aer_cfg.darts_fair_label,
    aer_cfg.darts_random_label,
)
title_suffix = " (Weber)"
filters = ("original_darts", "original_darts", "fair_darts", "random_darts")
figure_size = aer_cfg.figure_size
y_name = "validation loss"
y_label = aer_cfg.validation_loss_label
y_sem_name = "seed"
x1_name = "arch_weight_decay"
x1_label = aer_cfg.arch_weight_decay_label
x2_name = "num_graph_node"
# x2_name = 'num_params'
# x2_name = 'num_edges'
x2_label = aer_cfg.num_graph_nodes_label
x_limit = [-0.005, 0.105]
y_limit = [-0.005, 0.25]
arch_samp_filter = None

# y_name = 'num_params'
# y_label = 'Params'
# y_limit = [0, 12]


BIC = participant.compute_BIC(object_of_study=study_object_validation, num_params=0)
y_reference = None
y_reference_label = "Data Generating Model"

# DELLA
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_6.0_sample2_1"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_6.0_sample2_1"

# FINAL
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_4.0_sample0_3"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_4.0_sample0_3"

# ICML
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_2.0_sample0_3"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_2.0_sample0_3"


best_model_name = arch_weights_name

generate_darts_summary_figures(
    figure_names,
    titles,
    filters,
    title_suffix,
    study_name,
    y_name,
    y_label,
    y_sem_name,
    x1_name,
    x1_label,
    x2_name,
    x2_label,
    x_limit,
    y_limit,
    best_model_name,
    figure_size,
    y_reference,
    y_reference_label,
    arch_samp_filter,
)


# PLOT MODEL SEARCH RESULTS

figure_name = aer_cfg.figure_name_graph
plot_model_graph(
    study_name, arch_weights_name, model_weights_name, study_object, figure_name
)

# PLOT MODEL SIMULATION

model = load_model(study_name, model_weights_name, arch_weights_name, study_object)

figures_path = (
    aer_cfg.studies_folder
    + study_name
    + "/"
    + aer_cfg.models_folder
    + aer_cfg.models_results_figures_folder
)

participant.figure_plot(
    model,
    S1_list=(1, 2.5, 4),
    max_diff=5,
    num_data_points=100,
    figures_path=figures_path,
    figure_name=aer_cfg.figure_name_model_plot,
    figure_dimensions=aer_cfg.figure_size_model_plot,
    axis_font_size=aer_cfg.axis_font_size,
    title_font_size=aer_cfg.title_font_size,
    y_limit=(-0.3, 1.2),
    legend_font_size=aer_cfg.legend_font_size,
)
