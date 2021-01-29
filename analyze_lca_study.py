from AER_theorist.darts.plot_utils import plot_darts_summary, plot_model_graph
from AER_experimentalist.experiment_environment.participant_lca import Participant_LCA
from AER_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from AER_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from AER_experimentalist.experiment_environment.variable import outputTypes as output_type
import AER_experimentalist.experiment_environment.experiment_config as exp_cfg
import AER_config as aer_cfg
from AER_theorist.object_of_study import Object_Of_Study

participant = Participant_LCA()

study_name = "LCA Della"   # name of experiment
host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

# OBJECT OF STUDY

# specify independent variables
x1 = IV(name='x1_lca',
                          value_range=(-1, 1),
                          units="net input",
                          variable_label='x1')

x2 = IV(name='x2_lca',
                          value_range=(-1, 1),
                          units="net input",
                          variable_label='x2')

x3 = IV(name='x3_lca',
                          value_range=(-1, 1),
                          units="net input",
                          variable_label='x3')


# specify dependent variable with type
dx1_lca = DV(name='dx1_lca',
                          value_range=(0, 1),
                          units="net input change",
                          variable_label='dx1',
                          type=output_type.REAL) # not a probability because sum of activations may exceed 1


# list dependent and independent variables
IVs = [x1, x2, x3] # only including subset of available variables
DVs = [dx1_lca]

study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

### PLOT MODEL SEARCH RESULTS

figure_names = ('model_search_original', 'model_search_fair')
titles = (aer_cfg.darts_original_label, aer_cfg.darts_fair_label)
filters = ('original_darts', 'fair_darts')
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
y_limit = None

# for non-BIC models: choose model with best fit and num_params=num params of original model

# arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.25_k_2_s_8.0_sample2_1"
# model_weights_name = "model_weights_original_darts_v_1_wd_0.25_k_2_s_8.0_sample2_1"

# best log loss with num params (3) < num original params (4)
arch_weights_name = "architecture_weights_original_darts_v_1_wd_0.75_k_2_s_9.0_sample1_3"
model_weights_name = "model_weights_original_darts_v_1_wd_0.75_k_2_s_9.0_sample1_3"

# best log loss
# arch_weights_name = "architecture_weights_original_darts_v_1_wd_0_k_2_s_1.0_sample2_0"
# model_weights_name = "model_weights_original_darts_v_1_wd_0_k_2_s_1.0_sample2_0"

best_model_name = arch_weights_name

for idx, (figure_name, title, theorist_filter) in enumerate(zip(figure_names, titles, filters)):

    plot_darts_summary(study_name=study_name,
                       title=title,
                       y_name=y_name,
                       y_label=y_label,
                       y_sem_name=y_sem_name,
                       x1_name=x1_name,
                       x1_label = x1_label,
                       x2_name=x2_name,
                       x2_label = x2_label,
                       metric='mean_min',
                       x_limit=x_limit,
                       y_limit=y_limit,
                       best_model_name=best_model_name,
                       theorist_filter=theorist_filter,
                       figure_name = figure_name,
                       figure_dimensions=figure_size,
                       legend_loc=aer_cfg.legend_loc,
                       legend_font_size=aer_cfg.legend_font_size,
                       save=True)




### PLOT MODEL SEARCH RESULTS



figure_name = aer_cfg.figure_name_graph
plot_model_graph(study_name, arch_weights_name, model_weights_name, study_object, figure_name)