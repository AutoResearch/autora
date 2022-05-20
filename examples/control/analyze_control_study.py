from aer.theorist.darts.plot_utils import generate_darts_summary_figures, plot_model_graph, load_model
from aer.experiment_environment.participant_stroop import Participant_Stroop
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.variable.IV_in_silico import IV_In_Silico as IV
from aer.variable.DV_in_silico import DV_In_Silico as DV
from aer.variable.variable import outputTypes as output_type
from aer.experiment_environment import experiment_config as exp_cfg
from aer import config as aer_cfg
from aer.object_of_study import Object_Of_Study

participant = Participant_Stroop()

study_name = "Control Final 2" # "Control Della 2"   # name of experiment
study_name_sampled = "Control Della Sampled 2"   # name of experiment
host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

# OBJECT OF STUDY

# specify independent variables
color_red = IV(name='color_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Color Unit Red')

color_green = IV(name='color_green',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Color Unit Green')

task_color = IV(name='task_color',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Color Naming')


# specify dependent variable with type for validation set

verbal_red = DV(name='verbal_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Response Unit Red',
                          type=output_type.SIGMOID) # not a probability because sum of activations may exceed 1

verbal_green = DV(name='verbal_green',
                      value_range=(0, 1),
                      units="activation",
                      variable_label='Response Unit Green',
                      type=output_type.SIGMOID) # not a probability because sum of activations may exceed 1

verbal_sample = DV(name='verbal_sample',
                      value_range=(0, 1),
                      units="class",
                      variable_label='Verbal Response Sample',
                      type=output_type.CLASS)

# list dependent and independent variables
IVs = [color_red, color_green, task_color] # only including subset of available variables
DVs = [verbal_red, verbal_green]
DVs_validation = [verbal_sample]

study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

study_object_validation = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs_validation)

experimentalist_validation = Experimentalist_Popper(study_name=study_name_sampled,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

validation_data = experimentalist_validation.seed(study_object_validation, datafile='experiment_0_data.csv') # seed with new experiment
study_object_validation.add_data(validation_data)

### PLOT MODEL SEARCH RESULTS

figure_names = ('legend', 'model_search_original', 'model_search_random')
titles = (aer_cfg.darts_original_label, aer_cfg.darts_original_label, aer_cfg.darts_random_label)
title_suffix = " (Control)"
filters = ('original_darts', 'original_darts', 'random_darts')
figure_size = aer_cfg.figure_size
y_name = 'validation loss'
y_label = aer_cfg.validation_loss_label
y_sem_name = 'seed'
x1_name = 'arch_weight_decay'
x1_label = aer_cfg.arch_weight_decay_label
x2_name = 'num_graph_node'
# x2_name = 'num_params'
# x2_name = 'num_edges'
x2_label = aer_cfg.num_graph_nodes_label
x_limit = [-0.05, 1.05]
y_limit = None # [620, 840]

BIC = participant.compute_BIC(object_of_study=study_object_validation, num_params=4)
y_reference=None
y_reference_label='Data Generating Model'
# y_reference = None

# for non-BIC models: choose model with lowest BIC
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.75_k_3_s_1.0_sample0_0"
model_weights_name = "model_weights_original_darts_v_1_wd_0.75_k_3_s_1.0_sample0_0"

### FINAL:
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_2_s_1.0_sample0_1"
model_weights_name = "model_weights_original_darts_v_1_wd_0_k_2_s_1.0_sample0_1"

best_model_name = arch_weights_name

generate_darts_summary_figures(figure_names,
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

### PLOT MODEL SEARCH RESULTS

figure_name = aer_cfg.figure_name_graph
plot_model_graph(study_name, arch_weights_name, model_weights_name, study_object, figure_name)

### PLOT MODEL SIMULATION

model = load_model(study_name, model_weights_name, arch_weights_name, study_object)

figures_path = aer_cfg.studies_folder \
                   + study_name + "/" \
                   + aer_cfg.models_folder \
                   + aer_cfg.models_results_figures_folder

participant.figure_control_plot(model,
                            color_green_list=(0, 0, 1),
                            task_color_list=(0, 1, 1),
                            num_data_points=100,
                            figures_path=figures_path,
                            figure_name=aer_cfg.figure_name_model_plot,
                            figure_dimensions=aer_cfg.figure_size_model_plot,
                            y_limit=(0, 1),
                            legend_font_size=aer_cfg.legend_font_size)