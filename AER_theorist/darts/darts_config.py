import numpy as np
from enum import Enum

class InitMethod(Enum):
    UNIFORM = 1
    NORMAL = 2

# meta-search
max_k = 1                       # maximum number of computations (nodes) in model
min_k = 1                       # minimum number of computations (nodes) in model
min_seed = 1                    # minimum seed tested
max_seed = 1                    # maximum seed tested

# training
epochs = 5 #50 #100                    # num of training epochs (30)
arch_updates_per_epoch = 5 #20      # num of architecture updates per epoch (20)
param_updates_per_epoch = 500    # num of weight updates per epoch (20)
batch_size = 20                 # batch size (64)
learning_rate = 0.5           # init learning rate 0.025
learning_rate_min = 0.001 # 0.001       # min learning rate 0.001
momentum = 0.9                  # momentum 0.9
weight_decay = 3e-4 #3e-4             # weight decay 3e-4
seed = 1                        # random seed
grad_clip = 5                   # gradient clipping
train_portion = 0.8             # portion of training data
unrolled = False                # use one-step unrolled validation loss
arch_learning_rate = 0.3 #3e-4       # learning rate for arch encoding
arch_weight_decay = 1e-4 #1e-3        # general weight decay for arch encoding
bic_test_size = 100             # sample size for computing bic and aic
classifier_weight_decay = 1e-2  # L1 weight decay applied to classifier weights
custom_initialization = False
init_method = InitMethod.UNIFORM # method used to initialize weights during architecture evaluation (if re-initialize weights is set to True)
init_uniform_interval = [-10, 10]     # interval for uniform parameter initialization
init_normal_mean = 0                # mean for normal parameter initialization
init_normal_std = 1                  # standard deviation for normal parameter initialization

# evaluation
n_architectures_sampled = 1 #5      # num of (distinct) model architectures sampled from final architecture weights
n_initializations_sampled = 1 #20            # num of model initializations searched
max_arch_search_attempts = 100000
sample_amp = 100                # amplifies architecture weight differences before passing through softmax
reinitialize_weights = True     # whether to train sampled models on novel weights
eval_learning_rate = 0.3
eval_learning_rate_min = 0.01
eval_momentum = 0.0
eval_weight_decay = 0
eval_epochs = 3000 #5000
eval_custom_initialization = True


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
arch_weight_decay_list = [0] # 6e-4 # np.linspace(0, 6e-4, 6)               # list of architecture weight decays searched over (this decay scales with number of degrees of freedom)
num_node_list = np.linspace(min_k, max_k, (max_k-min_k)+1)     # list of k's searched over
seed_list = np.linspace(1, max_seed, max_seed)                 # list of seeds searched over
csv_model_file_name = 'model_file_name'
csv_arch_file_name = 'arch_file_name'
csv_num_graph_node = 'num_graph_node'
csv_log_loss = 'log_loss'