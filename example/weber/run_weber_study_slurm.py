import argparse
import os
import sys
from datetime import datetime

from aer.experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from example.weber.weber_setup import gen_params, study_object, validation_object_1

print(os.getcwd())
sys.path.append(r"/tigress/musslick/AER/cogsci2021")

# %%
# Parse arguments

parser = argparse.ArgumentParser("parser")
parser.add_argument("--slurm_id", type=int, default=1, help="number of slurm array")
args = parser.parse_args()

# %%
# Note current time

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# %%
# EXPERIMENTALIST

# Experiment design
stimulus_resolution = 20
weber_design = Experiment_Design_Synthetic_Weber(stimulus_resolution)

stimulus_resolution_validation = 100
weber_design_validation = Experiment_Design_Synthetic_Weber(
    stimulus_resolution_validation
)

# Initialize experimentalist
experimentalist = Experimentalist_Popper(
    study_name=gen_params.study_name,
    experiment_server_host=gen_params.host,
    experiment_server_port=gen_params.port,
    experiment_design=weber_design,
)

experimentalist_validation = Experimentalist_Popper(
    study_name=gen_params.study_name_sampled,
    experiment_server_host=gen_params.host,
    experiment_server_port=gen_params.port,
    experiment_design=weber_design_validation,
)

# %%
# THEORIST

# Initialize theorist
theorist = Theorist_DARTS(gen_params.study_name, darts_type=DARTS_Type.ORIGINAL)

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
theorist_fair = Theorist_DARTS(gen_params.study_name, darts_type=DARTS_Type.FAIR)
theorist_fair.plot()
theorist_fair.add_validation_set(validation_object_1, "Weber_Sampled")
theorist_fair.add_validation_set(validation_object_2, "Weber_Original")
model = theorist_fair.search_model_job(study_object, args.slurm_id)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
