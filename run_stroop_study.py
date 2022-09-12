# import os
# path_list = os.getcwd()
# print(os.listdir(path_list))

from datetime import datetime
from AER_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from AER_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from AER_experimentalist.experiment_environment.variable import outputTypes as output_type
from AER_experimentalist.experimentalist_popper import Experimentalist_Popper
from AER_experimentalist.experimentalist import Experimentalist
from AER_theorist.object_of_study import Object_Of_Study
from AER_theorist.theorist_darts import Theorist_DARTS, DARTS_Type
import AER_experimentalist.experiment_environment.experiment_config as exp_cfg

from AER_experimentalist.experimentalist_uncertainty_sampling import Experimentalist_Uncertainty_Sampling
from AER_experimentalist.experimentalist_random_sampling import Experimentalist_Random_Sampling



from AER_experimentalist.experiment_environment.experiment_server import Experiment_Server
from AER_experimentalist.experiment_environment.experiment_in_silico import Experiment_In_Silico
from AER_experimentalist.experiment_environment.participant_stroop import Participant_Stroop
import AER_experimentalist.experiment_environment.experiment_config as cfg


participant = Participant_Stroop() # the Cohen et al. (1990) Stroop Model
experiment = Experiment_In_Silico(participant=participant, main_directory=cfg.server_path)

# launch server
# server = Experiment_Server(host=cfg.HOST_IP, port=cfg.HOST_PORT, exp=experiment)
# server.launch()



now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

# SIMULATION PARAMETERS

study_name = "Stroop"   # name of experiment
study_name_sampled = "Stroop Sampled"   # name of experiment
max_num_data_points = 5000
max_num_data_points_sampled = 5000

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
# verbal_red_sampled = DV(name='verbal_red_sample',
#                           value_range=(0, 1),
#                           units="activation",
#                           variable_label='Response Unit Red',
#                           type=output_type.CLASS) # not a probability because sum of activations may exceed 1

# verbal_sample = DV(name='verbal_sample',
#                       value_range=(0, 1),
#                       units="class",
#                       variable_label='Verbal Response Sample',
#                       type=output_type.CLASS)


# list dependent and independent variables
IVs = [color_red, color_green, word_red, word_green, task_color] # only including subset of available variables
DVs = [verbal_red, verbal_green]
# DVs_validation = [verbal_sample]

# initialize objects of study
study_object = Object_Of_Study(name=study_name,
															 independent_variables=IVs,
															 dependent_variables=DVs)

# validation_object_1 = Object_Of_Study(name=study_name_sampled,
#                                independent_variables=IVs,
#                                dependent_variables=DVs_validation)

# EXPERIMENTALIST

# initialize experimentalist
# experimentalist = Experimentalist_Uncertainty_Sampling(study_name=study_name,
# 																	experiment_server_host=host,
# 																	experiment_server_port=port,
# 																	ivs=IVs)
experimentalist = Experimentalist_Random_Sampling(study_name=study_name,
																	experiment_server_host=host,
																	experiment_server_port=port,
																	ivs=IVs)


# experimentalist_validation = Experimentalist_Popper(study_name=study_name_sampled,
#                                   experiment_server_host=host,
#                                   experiment_server_port=port,
#                                          )

# THEORIST

# initialize theorist
theorist = Theorist_DARTS(study_name, darts_type=DARTS_Type.ORIGINAL)

# specify plots
# plots = list()
# plots.append(theorist._loss_plot_name)
# for i in range(20):
# 		plot_name = "Edge " + str(i)
# 		plots.append(plot_name)
# theorist.plot(plots)

# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1, n=5000) # seed with new experiment
# validation_data = experimentalist_validation.seed(validation_object_1, datafile='experiment_0_data.csv') # seed with new experiment
# validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
seed_data = experimentalist.seed(study_object, n=5000) # seed with new experiment
# seed_data = experimentalist.seed(study_object, datafile='experiment_0_data.csv') # seed with existing data file
print(seed_data.keys())
study_object.add_data(seed_data)

(input_data, output_data) = study_object.get_dataset()  
print('input_data shape:', input_data.shape)
print('input_data ex:', input_data[0, :]) 
print('output_data ex:', output_data[0, :]) 

# validation_object_2 = study_object.split(proportion=0.5)
# validation_object_2.name = "BIC"

# add validation sets
# theorist.add_validation_set(validation_object_1, 'BIC')
# theorist.add_validation_set(validation_object_2, 'validation loss')

AER_cycles = 20
for cycle in range(AER_cycles): 
	print('CURRENT AER CYCLE:', cycle) 

	# search model ORIGINAL
	model = theorist.search_model(study_object, cycle+1)

	# search model ORIGINAL
	# theorist_fair = Theorist_DARTS(study_name, darts_type=DARTS_Type.FAIR)
	# theorist_fair.plot(plots)
	# theorist_fair.add_validation_set(validation_object_1, 'BIC')
	# theorist_fair.add_validation_set(validation_object_2, 'validation loss')
	# model = theorist_fair.search_model(study_object)

	experiment_file_path = experimentalist.sample_experiment(model, study_object) 
	data = experimentalist.commission_experiment(study_object, experiment_file_path) 
	study_object.add_data(data) 

print('I am done !') 
# now = datetime.now()
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# print("date and time =", dt_string)