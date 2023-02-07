#!/bin/sh

#SBATCH -J TSExperimentalist
#SBATCH --time=46:00:00
#SBATCH --array=0-119
#SBATCH --mem=9GB
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
python run_CL_task_switching_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID
