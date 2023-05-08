from models.models import model_inventory
from sklearn.model_selection import train_test_split

from studies.cogsci2023.baseline_models.fleming import run_fleming

from random import seed
import numpy as np
import argparse
import pickle
import time

# META PARAMETERS
num_repetitions = 20                 # specifies how many times to repeat the study (20)
test_size = 0.2                # proportion of test set size to training set size
low_memory = True

bms_theorists = [
    'Root Fixed',
    'Root Variable',
    'Regular Fixed',
    'Regular Variable',
]

gts = ['fleming'
       ]

parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()

repetitions = np.arange(num_repetitions)
gt_ids = np.arange(len(gts))
conditions = np.array(np.meshgrid(repetitions, gt_ids)).T.reshape(-1,2)

rep = conditions[args.slurm_id, 0] # args.slurm_id
gt_id = conditions[args.slurm_id, 1]
ground_truth_name = gts[gt_id]
seed(rep)

for theorist in bms_theorists:
    run_fleming(rep, theorist)


