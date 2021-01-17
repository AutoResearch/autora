from AER_theorist.darts.plot_utils import plot_darts_summary


study_name = 'Stroop Model'
# y_name = 'loss'
# y_label = 'log loss'
y_name = 'Stroop_Sampled'
y_label = 'validaiton loss'
x1_name = 'arch_weight_decay'
x1_label = 'df penalty'
# x1_name = 'num_params'
# x1_label = 'df'
x2_name = 'num_graph_node'
x2_label = 'k'

# plot_darts_summary(study_name=study_name,
#                    y_name=y_name,
#                    x1_name=x1_name,
#                    y_label=y_label,
#                    x1_label=x1_label,
#                    metric='min')


plot_darts_summary(study_name=study_name,
                   y_name=y_name,
                   x1_name=x1_name,
                   x2_name=x2_name,
                   y_label=y_label,
                   x1_label=x1_label,
                   x2_label=x2_label,
                   metric='min')