from datetime import datetime
from tkinter import Tk

from aer.experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from aer.theorist.theorist_GUI import Theorist_GUI
from example.weber.weber_setup import gen_params, study_object, validation_object_1

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

# model = theorist.search_model(study_object)

root = Tk()
app = Theorist_GUI(object_of_study=study_object, theorist=theorist, root=root)
root.mainloop()

# root = Tk()
# app = AER_GUI(object_of_study=study_object, theorist=theorist,
# experimentalist=experimentalist, root=root)
# root.mainloop()

# theorist_fair = Theorist_DARTS(study_name, darts_type=DARTS_Type.FAIR)
# theorist_fair.plot()
# theorist_fair.add_validation_set(validation_object_1, 'Weber_Sampled')
# theorist_fair.add_validation_set(validation_object_2, 'Weber_Original')
# model = theorist_fair.search_model(study_object)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
