from aer.experimentalist.experiment_environment.variable import Variable as Var
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import Theorist_DARTS
from aer.gui import AER_GUI
from tkinter import *

# GENERAL PARAMETERS

study_name = "Simple Voltage"   # name of experiment
host = "192.168.188.27" # exp_env_cfg.HOST_IP      # ip address of experiment server
port = 47778 # exp_env_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variable
source_voltage = Var(name='source_voltage',
                          value_range=(0, 4000),
                          units="mV",
                          rescale = 0.0001,             # need to convert to V to keep input values small
                          variable_label='Source Voltage')

# specify dependent variable
target_voltage = Var(name='voltage0',
                          units="mV",
                          rescale = 0.0001,             # need to convert to V to keep input values small
                          variable_label='Target Voltage')

# list dependent and independent variables
IVs = [source_voltage]
DVs = [target_voltage]

# initialize object of study
study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimentalist_Popper(study_name=study_name,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                  seed_data_file="experiment_0_data.csv"
                                         )

# THEORIST
theorist = Theorist_DARTS(study_name)

# AUTONOMOUS EMPIRICAL RESEARCH

# seed_data = experimentalist.seed(study_object, n=20)
# study_object.add_data(seed_data)

# seed object of study with data
# seed_data = experimentalist.seed(study_object, datafile="experiment_0_data.csv")
# study_object.add_data(seed_data)
#
#
# root = Tk()
# app = Theorist_GUI(object_of_study=study_object, theorist=theorist, root=root)
# root.mainloop()

# theorist.GUI(study_object)


root = Tk()
app = AER_GUI(object_of_study=study_object, theorist=theorist, experimentalist=experimentalist, root=root)
root.mainloop()


# for cycle in range(AER_cycles):
#
#     # generate computational model to explain object of study
#     model = theorist.search_model(study_object)
#
#     # generate experiment based on the generated computational model and object of study
#     experiment_file_path = experimentalist.sample_experiment(model, study_object)
#
#     # collect data from experiment
#     data = experimentalist.commission_experiment(experiment_file_path)
#
#     # add new data to object of study
#     study_object.add_data(data)
#
# # TODO: for AER_class, make sure to log the cycle state, so that the cycle can be reinitiated at the point it was last interrupted