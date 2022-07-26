# %%
# Imports

import argparse

from aer.theorist.theorist_random_darts import Theorist_Random_DARTS
from example.weber.weber_setup import (
    experimentalist,
    experimentalist_validation,
    general_parameters,
    study_object,
    validation_object_1,
)

# todo:
# - import and instantiate theorist_random_darts
# - when instantiating, set theorist_filter to 'darts'
# - unplot

# %%
# Parse arguments

parser = argparse.ArgumentParser("parser")
parser.add_argument("--slurm_id", type=int, default=10, help="number of slurm array")
args = parser.parse_args()

# %%
# THEORIST

# Initialize theorist

theorist = Theorist_Random_DARTS(general_parameters.study_name, theorist_filter="darts")

# Specify plots

plots = list()
plots.append(theorist._loss_plot_name)
# theorist.plot(plot=True, plot_name_list=plots)

# %%
# AUTONOMOUS EMPIRICAL RESEARCH

# Generate first validation set

# validation_data = experimentalist_validation.seed(validation_object_1) # seed with new experiment
validation_data = experimentalist_validation.seed(
    validation_object_1, datafile="experiment_0_data.csv"
)  # seed with new experiment
validation_object_1.add_data(validation_data)

# Seed experiment and split into training/validation set

# seed_data = experimentalist.seed(study_object) # seed with new experiment
seed_data = experimentalist.seed(
    study_object, datafile="experiment_0_data.csv"
)  # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "validation loss"

# Add validation sets

theorist.add_validation_set(validation_object_1, "BIC")
theorist.add_validation_set(validation_object_2, "validation loss")

# %%
# ANALYSIS

# Search model

model = theorist.search_model_job(study_object, args.slurm_id)
