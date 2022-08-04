#!/bin/bash
#SBATCH -n 1      #up to the number of cores on a node - set in parpool below
#SBATCH -t 12:00:00
#SBATCH -J AER_Ctrl_Fair
#SBATCH --mail-type=end
#SBATCH --mail-user=musslick@princeton.edu
#SBATCH --export ALL
#SBATCH --mem=8000
#SBATCH --array=0-149

# anaconda env
source activate autora

python run_control_fair_study_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID
