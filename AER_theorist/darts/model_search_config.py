import numpy as np

# experiment
log_version = 1                 # simulation version (useful for keeping track of results from different runs)
object_of_study = 'SimpleNet'   # name of the data-generating object of study
draw_samples = False            # whether to randomly sample data from object of study
n_models_sampled = 1            # num of (distinct) model architectures sampled from final architecture weights
sample_amp = 100                # amplifies architecture weight differences before passing through softmax
reinitialize_weights = False     # whether to train sampled models on novel weights
max_k = 2                       # maximum number of computations (nodes) in model
min_k = 1                       # minimum number of computations (nodes) in model
max_seed = 5                    # maximum number of seeds tested
num_data_points = 10          # total number of queried data points from object of study

# training
epochs = 100                     # num of training epochs (30)
arch_updates_per_epoch = 5      # num of architecture updates per epoch (20)
param_updates_per_epoch = 5    # num of weight updates per epoch (20)
batch_size = 64                 # batch size (64)
learning_rate = 0.5           # init learning rate 0.025
learning_rate_min = 0.001       # min learning rate 0.001
momentum = 0.0                  # momentum 0.9
weight_decay = 3e-4             # weight decay 3e-4
seed = 1                        # random seed
grad_clip = 5                   # gradient clipping
train_portion = 0.5             # portion of training data
unrolled = False                # use one-step unrolled validation loss
arch_learning_rate = 3e-4       # learning rate for arch encoding
arch_weight_decay = 1e-3        # general weight decay for arch encoding
bic_test_size = 100             # sample size for computing bic and aic
classifier_weight_decay = 1e-2  # L1 weight decay applied to classifier weights

# hardware
gpu = 0                         # gpu device id

# logging and debugging
debug = True                                                   # debug mode
show_arch_weights = True                                      # flag whether to show weight updates in debug mode
report_freq = 5                                                # report frequency
model_path = 'saved_models'                                    # path to save the model
exp_folder = 'experiments'                                     # experiment name
graph_filename = 'model_graph'                                 # file name prefix of learned graph
model_filename = 'model_weights'                               # file name prefix of learned model
output_file_folder = 'results'                                 # folder for simulation results
arch_weight_decay_list = [6e-4] # np.linspace(0, 6e-4, 6)               # list of architecture weight decays searched over (this decay scales with number of degrees of freedom)
num_node_list = np.linspace(min_k, max_k, (max_k-min_k)+1)     # list of k's searched over
seed_list = np.linspace(1, max_seed, max_seed)                 # list of seeds searched over