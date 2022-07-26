import argparse
import os
import sys

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

print(os.getcwd())
sys.path.append(r"/tigress/musslick/AER/cogsci2021")

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

theorist = Theorist_DARTS(general_params.study_name, darts_type=DARTS_Type.ORIGINAL)

# Specify plots

plots = list()
plots.append(theorist._loss_plot_name)
theorist.plot()

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
validation_object_2.name = "Weber Sampled"

# Add validation sets

theorist.add_validation_set(validation_object_1, "Weber_Sampled")
theorist.add_validation_set(validation_object_2, "Weber_Original")

# %%
# ANALYSIS

# Search model

model = theorist.search_model_job(study_object, args.slurm_id)

# Fair search
theorist_fair = Theorist_DARTS(general_params.study_name, darts_type=DARTS_Type.FAIR)
theorist_fair.plot()
theorist_fair.add_validation_set(validation_object_1, "Weber_Sampled")
theorist_fair.add_validation_set(validation_object_2, "Weber_Original")
model = theorist_fair.search_model_job(study_object, args.slurm_id)
