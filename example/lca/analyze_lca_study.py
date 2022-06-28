import aer.config as aer_cfg
import aer_experimentalist.experiment_environment.experiment_config as exp_cfg
from aer_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from aer_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from aer_experimentalist.experiment_environment.participant_lca import Participant_LCA
from aer_experimentalist.experiment_environment.variable import (
    outputTypes as output_type,
)
from aer.theorist.darts.plot_utils import (
    generate_darts_summary_figures,
    load_model,
    plot_model_graph,
)
from aer.theorist.object_of_study import Object_Of_Study

participant = Participant_LCA()

study_name = "LCA ICML"  # "LCA Final"   # name of experiment: LCA Della
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

# OBJECT OF STUDY

# specify independent variables
x1 = IV(name="x1_lca", value_range=(-1, 1), units="net input", variable_label="x_1")

x2 = IV(name="x2_lca", value_range=(-1, 1), units="net input", variable_label="x_2")

x3 = IV(name="x3_lca", value_range=(-1, 1), units="net input", variable_label="x_3")


# specify dependent variable with type
dx1_lca = DV(
    name="dx1_lca",
    value_range=(0, 1),
    units="net input change",
    variable_label="dx_1",
    type=output_type.REAL,
)  # not a probability because sum of activations may exceed 1


# list dependent and independent variables
IVs = [x1, x2, x3]  # only including subset of available variables
DVs = [dx1_lca]

study_object = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)

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
title_suffix = " (LCA)"
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
y_limit = [-0.005, 0.085]
arch_samp_filter = 0


# y_name = 'num_params'
# y_label = 'Params'
# y_limit = [0, 12]

# for non-BIC models: choose model with best fit and num_params<=num params of original model

# arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.25_k_2_s_8.0_sample2_1"
# model_weights_name = "model_weights_original_darts_v_1_wd_0.25_k_2_s_8.0_sample2_1"

# best log loss with num params (3) < num original params (4)
arch_weights_name = (
    "architecture_weights_original_darts_v_1_wd_0.75_k_2_s_9.0_sample0_1"
)
model_weights_name = "model_weights_original_darts_v_1_wd_0.75_k_2_s_9.0_sample0_1"

# model with lowest log loss
# DELLA
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample0_4"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample0_4"

# FINAL (k=1)
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample0_4"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample0_4"

# FINAL (k=3)
# arch_weights_name = "architecture_weights_original_darts_v_1_wd_1.0_k_3_s_8.0_sample0_4"
# model_weights_name = "model_weights_original_darts_v_1_wd_1.0_k_3_s_8.0_sample0_4"

# ICML
# arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.1_k_3_s_7.0_sample0_0"
# model_weights_name = "model_weights_original_darts_v_1_wd_0.1_k_3_s_7.0_sample0_0"

# ICML Final
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.1_k_1_s_2.0_sample0_4"
model_weights_name = "model_weights_original_darts_v_1_wd_0.1_k_1_s_2.0_sample0_4"

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
    arch_samp_filter=arch_samp_filter,
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
    x1=0.4,  # 0.3
    x2=0.6,  # 0.6
    x3=-0.2,  # -0.2
    n_trials=20,
    figures_path=figures_path,
    figure_name=aer_cfg.figure_name_model_plot,
    figure_dimensions=aer_cfg.figure_size_model_plot,
    y_limit=(-0.55, 1.2),
    axis_font_size=aer_cfg.axis_font_size,
    title_font_size=aer_cfg.title_font_size,
    legend_font_size=aer_cfg.legend_font_size,
)
