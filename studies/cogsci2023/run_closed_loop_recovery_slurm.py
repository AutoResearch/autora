import pickle
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from models.models import model_inventory
from utils import (
    fit_theorist,
    get_experimentalist,
    get_MSE,
    get_seed_experimentalist,
)
from run_closed_loop_recovery import closed_loop

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()
rep = args.slurm_id

# META PARAMETERS
num_cycles = 20  # number of cycles (20)
samples_for_seed = 10  # number of seed data_closed_loop points (20)
samples_per_cycle = 10  # number of data_closed_loop points chosen per cycle (20)
theorist_epochs = 1500  # number of epochs for BMS (500)

# SELECT THEORIST
# OPTIONS: BMS, DARTS
theorist_name = "BMS"

# SELECT GROUND TRUTH MODEL
ground_truth_name = "prospect_theory"  # OPTIONS: see models.py

experimentalists = [
    # 'popper',
    # 'falsification',
    'random',
    # "dissimilarity",
    # 'model disagreement',
    # 'least confident',
]

st = time.time()
print("hello world")
# closed_loop(rep,
#             theorist_name,
#             ground_truth_name,
#             experimentalists,
#             num_cycles,
#             samples_for_seed,
#             samples_per_cycle,
#             theorist_epochs)
et = time.time()
elapsed_time = et - st
print(f"Elapsed time: {elapsed_time}")
