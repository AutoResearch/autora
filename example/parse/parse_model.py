import argparse
import os
import re

import torch.nn as nn
import torch.utils
from torch.autograd import Variable

try:
    import cnnsimple.model_search_config as cfg
    import cnnsimple.utils as utils
    import cnnsimple.visualize as viz
    from cnnsimple.model_search import Network

except ImportError:
    # MyPy throws errors when looking at these imports.
    # This is a known problem: https://github.com/python/mypy/issues/1153
    # Fix: include "# type: ignore" comments on each import
    # Also occurs in run_model_search.py
    import aer.theorist.darts.model_search_config as cfg  # type: ignore
    import aer.theorist.darts.utils as utils  # type: ignore
    import aer.theorist.darts.visualize as viz  # type: ignore
    from aer.theorist.darts.model_search import Network  # type: ignore


# PARSE ARGUMENTS

parser = argparse.ArgumentParser("modelSearch")
parser.add_argument(
    "--object_of_study",
    type=str,
    default=cfg.object_of_study,
    help="name of data generating object of study; available options: 'SimpleNet' ",
)
parser.add_argument(
    "--log_version", type=int, default=cfg.log_version, help="log version"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="model_weights_v_1_wd_0.00052_k_1_s_1_sample0",
    help="name of model to parse",
)

args = parser.parse_args()

args.obj_of_study_file = utils.get_object_of_study_file(args.object_of_study)
args.obj_of_study_class = utils.get_object_of_study(args.object_of_study)
args.obj_of_study = args.obj_of_study_class(
    num_patterns=cfg.num_data_points, sampling=cfg.draw_samples
)
args.inputDim = args.obj_of_study.input_dimensions
args.outputDim = args.obj_of_study.output_dimensions
args.outputType = args.obj_of_study.output_type
args.loss = utils.get_loss_function(args.outputType)
args.input_labels = args.obj_of_study.input_labels

args.model_path = cfg.model_path
args.exp_folder = cfg.exp_folder
args.output_file_folder = cfg.output_file_folder

# define loss function
criterion = utils.get_loss_function(args.obj_of_study.output_type)

# DETERMINE LOG FOLDER

print("object_of_study: " + args.object_of_study)
print("log_version: " + str(args.log_version))

args.save = "{}-v{}".format(args.obj_of_study.__get_name__(), str(args.log_version))
args.arch_name = args.model_name.replace("model_", "arch_")

model_path = os.path.join(
    args.exp_folder, args.save, args.output_file_folder, (args.model_name + ".pt")
)
arch_path = os.path.join(
    args.exp_folder, args.save, args.output_file_folder, (args.arch_name + ".pt")
)

# LOAD MODEL

# determine number of nodes from model name

search_result = re.search("k_(.+?)_s", args.model_name)

if search_result is not None:
    found = search_result.group(1)

else:
    found = ""


args.num_graph_nodes = int(found)

print("Loading model...")
print(model_path)

model = Network(
    args.outputDim,
    criterion,
    steps=int(args.num_graph_nodes),
    n_input_states=int(args.inputDim),
)
utils.load(model, model_path)
alphas_normal = torch.load(arch_path)
model.fix_architecture(True, new_weights=alphas_normal)

# PARSING MODEL

print("Parsing model...")

# save model plot
genotype = model.genotype()
(n_params_total, n_params_base, param_list) = model.countParameters(
    print_parameters=True
)
viz.plot(
    genotype.normal,
    "tmp",
    fileFormat="pdf",
    full_label=True,
    param_list=param_list,
    input_labels=args.input_labels,
)


# validate model
stimulus = Variable(torch.FloatTensor([[0.3, 0.8]]), requires_grad=False)
out = model(stimulus)
print("input:")
print(stimulus)

print("output:")
print(out)

# validate BIC computation
soft_target = args.obj_of_study_file.get_target(stimulus).data.numpy()
softmax = nn.Softmax(dim=1)
soft_prediction = softmax(model(stimulus)).data.numpy()

BIC, AIC = utils.compute_BIC_AIC(soft_target, soft_prediction, model)

# TODO

# (x) write method in run_model_search that also stores architecture weights
# (x) write method to generate architecture weights from genotype
# (x) then fix architecture weights in model below
# (x) visualize complete architecture
# (x) validate computation
# (x) validate BIC computation for 2 data points
# (x) implement lasso (L1) regularization for classifier weights
#     (to select as few features as possible that feed to output layer)
# (x) implement Stroop model
# (x) validate Stroop model
# (x) implement Stroop data set
#  - architecture search Stroop model
# (-) rewrite the whole data generator thing (for creating plot variables and such)
