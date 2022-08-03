from datetime import datetime
from tkinter import Tk

import aer.experimentalist.experiment_environment.experiment_config as exp_cfg
from aer.experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import DARTS_Type, Theorist_DARTS
from aer.theorist.theorist_GUI import Theorist_GUI
from aer.variable import DVInSilico as DV
from aer.variable import IVInSilico as IV
from aer.variable import ValueType as output_type

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

study_name = "Weber"  # name of experiment
study_name_sampled = "Weber Sampled"
host = exp_cfg.HOST_IP  # ip address of experiment server
port = exp_cfg.HOST_PORT  # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
S1 = IV(
    name="S1",
    value_range=(0, 5),
    units="intensity",
    variable_label="Stimulus 1 Intensity",
)

S2 = IV(
    name="S2",
    value_range=(0, 5),
    units="intensity",
    variable_label="Stimulus 2 Intensity",
)


# specify dependent variable with type
diff_detected = DV(
    name="difference_detected",
    value_range=(0, 1),
    units="probability",
    variable_label="P(difference detected)",
    type=output_type.SIGMOID,
)

diff_detected_sample = DV(
    name="difference_detected_sample",
    value_range=(0, 1),
    units="response",
    variable_label="difference detected",
    type=output_type.PROBABILITY_SAMPLE,
)

# list dependent and independent variables
IVs = [S1, S2]  # only including subset of available variables
DVs = [diff_detected]
DVs_validation = [diff_detected_sample]

study_object = Object_Of_Study(
    name=study_name, independent_variables=IVs, dependent_variables=DVs
)
# initialize objects of study

validation_object_1 = Object_Of_Study(
    name=study_name_sampled,
    independent_variables=IVs,
    dependent_variables=DVs_validation,
)

# EXPERIMENTALIST

# experiment design
stimulus_resolution = 20
weber_design = Experiment_Design_Synthetic_Weber(stimulus_resolution)

stimulus_resolution_validation = 100
weber_design_validation = Experiment_Design_Synthetic_Weber(
    stimulus_resolution_validation
)

# initialize experimentalist
experimentalist = Experimentalist_Popper(
    study_name=study_name,
    experiment_server_host=host,
    experiment_server_port=port,
    experiment_design=weber_design,
)

experimentalist_validation = Experimentalist_Popper(
    study_name=study_name_sampled,
    experiment_server_host=host,
    experiment_server_port=port,
    experiment_design=weber_design_validation,
)

# THEORIST

# initialize theorist
theorist = Theorist_DARTS(
    study_name, darts_type=DARTS_Type.ORIGINAL, primitives=["none", "add", "subtract"]
)

# specify plots
plots = list()
plots.append(theorist._loss_plot_name)
theorist.plot()


# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1) # seed with new experiment
validation_data = experimentalist_validation.seed(
    validation_object_1, datafile="experiment_0_data.csv"
)  # seed with new experiment
validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object) # seed with new experiment
seed_data = experimentalist.seed(
    study_object, datafile="experiment_0_data.csv"
)  # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "Weber Sampled"

# add validation sets
theorist.add_validation_set(validation_object_1, "Weber_Sampled")
theorist.add_validation_set(validation_object_2, "Weber_Original")

# search model
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
