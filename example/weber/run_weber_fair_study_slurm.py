# %%
# Imports

import argparse
from typing import List

from aer.experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from example.weber.weber_setup import (
    experimentalist,
    experimentalist_validation,
    general_params,
    study_object,
    validation_object_1,
)

# %%
# Common setup


# %%
# Parse arguments

parser = argparse.ArgumentParser("parser")
parser.add_argument("--slurm_id", type=int, default=1, help="number of slurm array")
args = parser.parse_args()

# %%
# EXPERIMENTALIST

# Experiment design

stimulus_resolution = 20
weber_design = Experiment_Design_Synthetic_Weber(stimulus_resolution)

stimulus_resolution_validation = 100
weber_design_validation = Experiment_Design_Synthetic_Weber(
    stimulus_resolution_validation
)

# %%
# THEORIST

# Initialize theorist

theorist = Theorist_DARTS(general_params.study_name, darts_type=DARTS_Type.FAIR)

# Specify plots

plots: List[str] = list()
# plots.append(theorist._loss_plot_name)
# for i in range(20):
#     plot_name = "Edge " + str(i)
#     plots.append(plot_name)
theorist.plot(plot=False, plot_name_list=plots)

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
