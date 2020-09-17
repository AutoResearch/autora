import os
import sys
import time
import glob
import copy
import csv
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms

from torch.autograd import Variable

try:
    import cnnsimple.utils as utils
    import cnnsimple.visualize as viz
    import cnnsimple.plot_utils as plotutils
    import cnnsimple.genotypes
    from cnnsimple.model_search import Network
    from cnnsimple.architect import Architect
    import cnnsimple.model_search_config as cfg
    from cnnsimple.object_of_study import outputTypes
    from cnnsimple.SimpleNet_dataset import SimpleNetDataset
except:
    import utils as utils
    import visualize as viz
    import plot_utils as plotutils
    import genotypes
    from model_search import Network
    from architect import Architect
    import model_search_config as cfg
    from object_of_study import outputTypes
    from SimpleNet_dataset import SimpleNetDataset

# PARSE ARGUMENTS

parser = argparse.ArgumentParser("modelSearch")
parser.add_argument('--object_of_study', type=str, default=cfg.object_of_study, help='name of data generating object of study; available options: \'SimpleNet\' ')
parser.add_argument('--log_version', type=int, default=cfg.log_version, help='log version')
parser.add_argument('--topk', type=int, default=5, help='number of best-fitting models listed')
parser.add_argument('--winning_architecture_only', dest='winning_arch_only', action='store_true', help='only search models with highest architecture weights')
parser.set_defaults(winning_arch_only=False)

args = parser.parse_args()

args.obj_of_study_class = utils.get_object_of_study(args.object_of_study)
args.obj_of_study = args.obj_of_study_class(num_patterns=cfg.num_data_points, sampling=cfg.draw_samples)
args.model_path = cfg.model_path
args.exp_folder = cfg.exp_folder
args.output_file_folder = cfg.output_file_folder

# DETERMINE LOG FOLDER

print('object_of_study: ' + args.object_of_study)
print('log_version: ' + str(args.log_version))

args.save = '{}-v{}'.format(args.obj_of_study.__get_name__(), str(args.log_version))
resultsPath = os.path.join(args.exp_folder, args.save, args.output_file_folder)

(model_name_list, loss_list, BIC_list, AIC_list) = utils.read_log_files(resultsPath, args.winning_arch_only)

# IDENTIFY BEST FITTING MODELS

(topk_loss_names, topk_BIC_names) = utils.get_best_fitting_models(model_name_list, loss_list, BIC_list, args.topk)

print('Models with lowest loss:')
print('========================')
for name in topk_loss_names:
    print(name)
print('------------------------')


print('Models with lowest BIC:')
print('========================')
for name in topk_BIC_names:
    print(name)