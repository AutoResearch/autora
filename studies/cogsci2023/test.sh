#!/bin/sh

#SBATCH -J Test
#SBATCH --time=23:00:00
#SBATCH --mem=4GB
#SBATCH -n 1
#SBATCH --mail-user=sebastian_musslick@brown.edu

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
python test.py
