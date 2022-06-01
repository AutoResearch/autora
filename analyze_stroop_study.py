import time

import AER_experimentalist.experiment_environment.experiment_config as exp_cfg
from AER_experimentalist.experiment_environment.DV_in_silico import \
    DV_In_Silico as DV
from AER_experimentalist.experiment_environment.IV_in_silico import \
    IV_In_Silico as IV
from AER_experimentalist.experiment_environment.participant_stroop import \
    Participant_Stroop
from AER_experimentalist.experiment_environment.variable import \
    outputTypes as output_type
from AER_experimentalist.experimentalist_popper import Experimentalist_Popper
from AER_theorist.darts.plot_utils import plot_darts_summary
from AER_theorist.object_of_study import Object_Of_Study

participant = Participant_Stroop()

study_name = "Stroop Della"   # name of experiment
study_name_sampled = "Stroop Della Sampled"   # name of experiment
host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

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

word_red = IV(name='word_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Word Unit Red')

word_green = IV(name='word_green',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Word Unit Green')

task_color = IV(name='task_color',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Color Naming')

task_word = IV(name='task_word',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Word Reading')

# specify dependent variable with type for validation set
verbal_red_sampled = DV(name='verbal_red_sample',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Response Unit Red',
                          type=output_type.CLASS) # not a probability because sum of activations may exceed 1

verbal_sample = DV(name='verbal_sample',
                      value_range=(0, 1),
                      units="class",
                      variable_label='Verbal Response Sample',
                      type=output_type.CLASS)

IVs = [color_red, color_green] # only including subset of available variables
DVs_validation = [verbal_sample]

validation_object_1 = Object_Of_Study(name=study_name_sampled,
                               independent_variables=IVs,
                               dependent_variables=DVs_validation)

experimentalist_validation = Experimentalist_Popper(study_name=study_name_sampled,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

validation_data = experimentalist_validation.seed(validation_object_1, datafile='experiment_0_data.csv') # seed with new experiment
validation_object_1.add_data(validation_data)

BIC = participant.compute_BIC(object_of_study=validation_object_1, num_params=21)
y_reference=BIC
y_reference_label='Data Generating Model'

# y_name = 'loss'
# y_label = 'log loss'
# y_name = 'validation loss'
y_name = 'BIC'

# x1_name = 'num_params'
x1_name = 'num_graph_node'
# x1_name = 'arch_weight_decay'
# x1_name = 'num_edges'

# x2_name = 'num_graph_node'
# x2_name = 'num_params'
x_limit = [0, 1]

plot_darts_summary(study_name=study_name,
                   y_name=y_name,
                   x1_name=x1_name,
                   metric='min',
                   y_reference=y_reference)


# plot_darts_summary(study_name=study_name,
#                    y_name=y_name,
#                    x1_name=x1_name,
#                    x2_name=x2_name,
#                    metric='min',
#                    x_limit=x_limit,
#                    theorist_filter='original_darts')


# plot_darts_summary(study_name=study_name,
#                    y_name=y_name,
#                    x1_name=x1_name,
#                    x2_name=x2_name,
#                    y_label=y_label,
#                    x1_label=x1_label,
#                    x2_label=x2_label,
#                    metric='mean',
#                    y_reference=y_reference,
#                    y_reference_label=y_reference_label)

# participant.graph_simple('test')

