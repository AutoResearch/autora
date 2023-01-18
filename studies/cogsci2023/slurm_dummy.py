print("hello world")

import argparse

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()
rep = args.slurm_id

print("hello world2")

