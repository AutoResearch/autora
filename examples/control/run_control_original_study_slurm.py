from datetime import datetime
from aer.variable.IV_in_silico import IV_In_Silico as IV
from aer.variable.DV_in_silico import DV_In_Silico as DV
from aer.variable.variable import outputTypes as output_type
from aer.experimentalist.experimentalist_popper import Experimentalist_Popper
from aer.theorist.object_of_study import Object_Of_Study
from aer.theorist.theorist_darts import Theorist_DARTS, DARTS_Type
from aer.experiment_environment import experiment_config as exp_cfg
import argparse

# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=1, help='number of slurm array')
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

# SIMULATION PARAMETERS

study_name = "Control Final" #"Control Della 2"   # name of experiment
study_name_sampled = "Control Final Sampled" # "Control Della Sampled 2"   # name of experiment
max_num_data_points = 550
max_num_data_points_sampled = 550

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

task_color = IV(name='task_color',
                          value_range=(0, 1),
                          units="activation",
                          variable_label='Task Unit Color Naming')



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
IVs = [color_red, color_green, task_color] # only including subset of available variables
DVs = [verbal_red, verbal_green]
DVs_validation = [verbal_sample]

# initialize objects of study
study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

validation_object_1 = Object_Of_Study(name=study_name_sampled,
                               independent_variables=IVs,
                               dependent_variables=DVs_validation)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimentalist_Popper(study_name=study_name,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

experimentalist_validation = Experimentalist_Popper(study_name=study_name_sampled,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                         )

# THEORIST

# initialize theorist
theorist = Theorist_DARTS(study_name, darts_type=DARTS_Type.ORIGINAL)

# specify plots
plots = list()
plots.append(theorist._loss_plot_name)
# for i in range(20):
#     plot_name = "Edge " + str(i)
#     plots.append(plot_name)
#
# theorist.plot(plot_name_list=plots)

# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1, n=max_num_data_points_sampled) # seed with new experiment
validation_data = experimentalist_validation.seed(validation_object_1, datafile='experiment_0_data.csv') # seed with new experiment
validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object, n=max_num_data_points) # seed with new experiment
seed_data = experimentalist.seed(study_object, datafile='experiment_0_data.csv') # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "BIC"

# add validation sets
theorist.add_validation_set(validation_object_1, 'BIC')
theorist.add_validation_set(validation_object_2, 'validation loss')

# search model ORIGINAL
model = theorist.search_model_job(study_object, args.slurm_id)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)