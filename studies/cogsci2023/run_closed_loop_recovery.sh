#!/bin/sh

#SBATCH -J Experimentalists
#SBATCH --time=1:00:00
#SBATCH --array=1-10
#SBATCH --mem=32GB
#SBATCH -n 1

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load graphviz/2.40.1
module load python/3.9.0
module load git/2.29.2
source ~/autora_dev/bin/activate
cd /users/smusslic/research/AER_repo/studies/cogsci2023

# Run job
python run_closed_loop_recovery_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID