# %%
# Import modules

from tkinter import Tk

from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from aer.theorist.theorist_GUI import Theorist_GUI
from example.weber.weber_setup import (
    experimentalist,
    experimentalist_validation,
    general_parameters,
    study_object,
    validation_object_1,
)

# %%
# Print setup objects

print(
    experimentalist,
    experimentalist_validation,
    general_parameters,
    study_object,
    validation_object_1,
)

# %%
# THEORIST

# Initialize theorist

theorist = Theorist_DARTS(general_parameters.study_name, darts_type=DARTS_Type.ORIGINAL)

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
