import numpy as np

import aer.config as aer_cfg
import aer.experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.experimentalist.experiment_environment.participant_exp_learning import (
    Participant_Exp_Learning,
)
from aer.object_of_study import ObjectOfStudy
from aer.theorist.darts.plot_utils import (
    generate_darts_summary_figures,
    load_model,
    plot_model_graph,
)
from aer.variable import DVInSilico as DV
from aer.variable import IVInSilico as IV
from aer.variable import ValueType as output_type

participant = Participant_Exp_Learning()

study_name = (
    "Exp Learning ICML"  # "Exp Learning 3"   # name of experiment: Exp Learning Della
)
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

# OBJECT OF STUDY

# specify independent variables
learning_trial = IV(
    name="learning_trial", value_range=(0, 1), units="trial", variable_label="t"
)

P_initial = IV(
    name="P_initial", value_range=(0, 0.4), units="accuracy", variable_label="P_0"
)

P_asymptotic = IV(
    name="P_asymptotic", value_range=(0.5, 1), units="accuracy", variable_label="P_inf"
)


# specify dependent variable with type
learning_performance = DV(
    name="learning_performance",
    value_range=(0, 1),
    units="probability",
    variable_label="Accuracy",
    type=output_type.REAL,
)


# list dependent and independent variables
IVs = [learning_trial, P_initial, P_asymptotic]
DVs = [learning_performance]

study_object = ObjectOfStudy(
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
title_suffix = " (Learning)"
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
y_limit = [0, 0.065]

y_reference = None
y_reference_label = None

# y_name = 'num_params'
# y_label = 'Params'
# y_limit = [0, 12]

# for non-BIC models: choose model with best fit and num_params<=num params of original model.
# If not available choose best model for num_params+1
# arch_weights_name = "architecture_weights_fair_darts_v_1_wd_0_k_2_s_10.0_sample2_3"
# model_weights_name = "model_weights_fair_darts_v_1_wd_0_k_2_s_10.0_sample2_3"

# DELLA
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample2_3"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_1_s_9.0_sample2_3"

# FINAL
arch_weights_name = (
    "architecture_weights_original_darts_v_1_wd_0.025_k_3_s_4.0_sample0_0"
)
model_weights_name = "model_weights_original_darts_v_1_wd_0.025_k_3_s_4.0_sample0_0"

# FINAL 2
# arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.25_k_1_s_3.0_sample0_1"
# model_weights_name = "model_weights_original_darts_v_1_wd_0.25_k_1_s_3.0_sample0_1"

# ICML
arch_weights_name = (
    "architecture_weights_original_darts_v_1_wd_0.025_k_3_s_4.0_sample0_0"
)
model_weights_name = "model_weights_original_darts_v_1_wd_0.025_k_3_s_4.0_sample0_0"


best_model_name = arch_weights_name  # arch_weights_name # arch_weights_name

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
    P_initial=(0, 0.25, 0.25),
    P_asymptotic=(1, 1, 0.75),
    learning_trials=np.linspace(0, 1, 10),
    figures_path=figures_path,
    figure_name=aer_cfg.figure_name_model_plot,
    figure_dimensions=aer_cfg.figure_size_model_plot,
    axis_font_size=aer_cfg.axis_font_size,
    title_font_size=aer_cfg.title_font_size,
    y_limit=(-0.3, 1.2),
    legend_font_size=aer_cfg.legend_font_size,
)
