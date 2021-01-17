from datetime import datetime
from AER_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from AER_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from AER_experimentalist.experiment_environment.variable import outputTypes as output_type
from AER_experimentalist.experimentalist_popper import Experimentalist_Popper
from AER_experimentalist.experimentalist import Experimentalist
from AER_theorist.object_of_study import Object_Of_Study
from AER_theorist.theorist_darts import Theorist_DARTS
import AER_experimentalist.experiment_environment.experiment_config as exp_cfg
from AER_theorist.theorist_GUI import Theorist_GUI
from AER_GUI import AER_GUI
from tkinter import *

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

study_name = "Stroop Model"   # name of experiment
host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
color_red = IV(name='color_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Color Unit Red')

color_green = IV(name='color_green',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Color Unit Green')

word_red = IV(name='word_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Word Unit Red')

word_green = IV(name='word_green',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Word Unit Green')

task_color = IV(name='task_color',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Color Naming')

task_word = IV(name='task_word',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Word Reading')



# specify dependent variable with type
verbal_red = DV(name='verbal_red',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Response Unit Red',
                          type=output_type.SIGMOID) # not a probability because sum of activations may exceed 1

verbal_green = DV(name='verbal_green',
                      value_range=(0, 1),
                      units="activation",
                      variable_label='Response Unit Green',
                      type=output_type.SIGMOID) # not a probability because sum of activations may exceed 1

# specify dependent variable with type for validation set
verbal_red_sampled = DV(name='verbal_red_sample',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Response Unit Red',
                          type=output_type.CLASS) # not a probability because sum of activations may exceed 1

verbal_sample = DV(name='verbal_sample',
                      value_range=(0, 1),
                      units="class",
                      variable_label='Verbal Response Sample',
                      type=output_type.CLASS)


# list dependent and independent variables
IVs = [color_red, color_green] # only including subset of available variables
DVs = [verbal_red, verbal_green]
DVs_validation = [verbal_sample]

# initialize objects of study
study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

validation_object_1 = Object_Of_Study(name="Stroop Model Sampled",
                               independent_variables=IVs,
                               dependent_variables=DVs_validation)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimentalist_Popper(study_name=study_name,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

experimentalist_validation = Experimentalist_Popper(study_name="Stroop Model Sampled",
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

# THEORIST

# initialize theorist
theorist = Theorist_DARTS(study_name)
theorist.plot = True

# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1, n=1000) # seed with new experiment
validation_data = experimentalist_validation.seed(validation_object_1, datafile='experiment_0_data.csv') # seed with new experiment
validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object, n=100) # seed with new experiment
seed_data = experimentalist.seed(study_object, datafile='experiment_0_data.csv') # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "Stroop Sampled"

# add validation sets
theorist.add_validation_set(validation_object_1, 'Stroop_Sampled')
theorist.add_validation_set(validation_object_2, 'Stroop_Original')

# search model
model = theorist.search_model(study_object)

# root = Tk()
# app = Theorist_GUI(object_of_study=study_object, theorist=theorist, root=root)
# root.mainloop()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)