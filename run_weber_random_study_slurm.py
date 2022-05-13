from datetime import datetime
from AER_experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from AER_experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from AER_experimentalist.experiment_environment.variable import outputTypes as output_type
from AER_experimentalist.experimentalist_popper import Experimentalist_Popper
from AER_experimentalist.experiment_design_synthetic_weber import Experiment_Design_Synthetic_Weber
from AER_theorist.object_of_study import Object_Of_Study
from AER_theorist.theorist_random_darts import Theorist_Random_DARTS
import AER_experimentalist.experiment_environment.experiment_config as exp_cfg
import argparse

# todo:
# - import and instantiate theorist_random_darts
# - when instantiating, set theorist_filter to 'darts'
# - unplot


# parse arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=10, help='number of slurm array')
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# GENERAL PARAMETERS

study_name = "Weber ICML"   # name of experiment
study_name_sampled = "Weber Sampled"
host = exp_cfg.HOST_IP      # ip address of experiment server
port = exp_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variables
S1 = IV(name='S1',
        value_range=(0, 5),
        units="intensity",
        variable_label='Stimulus 1 Intensity')

S2 = IV(name='S2',
        value_range=(0, 5),
        units="intensity",
        variable_label='Stimulus 2 Intensity')



# specify dependent variable with type
diff_detected = DV(name='difference_detected',
                          value_range=(0, 1),
                          units="probability",
                          variable_label='P(difference detected)',
                          type=output_type.SIGMOID)

diff_detected_sample = DV(name='difference_detected_sample',
                          value_range=(0, 1),
                          units="response",
                          variable_label='difference detected',
                          type=output_type.PROBABILITY_SAMPLE)

# list dependent and independent variables
IVs = [S1, S2] # only including subset of available variables
DVs = [diff_detected]
DVs_validation = [diff_detected_sample]

# initialize objects of study
study_object = Object_Of_Study(name=study_name,
                               independent_variables=IVs,
                               dependent_variables=DVs)

validation_object_1 = Object_Of_Study(name=study_name_sampled,
                               independent_variables=IVs,
                               dependent_variables=DVs_validation)

# EXPERIMENTALIST

# experiment design
stimulus_resolution = 20
weber_design = Experiment_Design_Synthetic_Weber(stimulus_resolution)

stimulus_resolution_validation = 100
weber_design_validation = Experiment_Design_Synthetic_Weber(stimulus_resolution_validation)

# initialize experimentalist
experimentalist = Experimentalist_Popper(study_name=study_name,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                  experiment_design=weber_design)

experimentalist_validation = Experimentalist_Popper(study_name=study_name_sampled,
                                  experiment_server_host=host,
                                  experiment_server_port=port,
                                  experiment_design=weber_design_validation)

# THEORIST

# initialize theorist
theorist = Theorist_Random_DARTS(study_name, theorist_filter='darts')

# specify plots
plots = list()
plots.append(theorist._loss_plot_name)
# theorist.plot(plot=True, plot_name_list=plots)

# AUTONOMOUS EMPIRICAL RESEARCH

# generate first validation set
# validation_data = experimentalist_validation.seed(validation_object_1) # seed with new experiment
validation_data = experimentalist_validation.seed(validation_object_1, datafile='experiment_0_data.csv') # seed with new experiment
validation_object_1.add_data(validation_data)

# seed experiment and split into training/validation set
# seed_data = experimentalist.seed(study_object) # seed with new experiment
seed_data = experimentalist.seed(study_object, datafile='experiment_0_data.csv') # seed with existing data file
study_object.add_data(seed_data)
validation_object_2 = study_object.split(proportion=0.5)
validation_object_2.name = "validation loss"

# add validation sets
theorist.add_validation_set(validation_object_1, 'BIC')
theorist.add_validation_set(validation_object_2, 'validation loss')

# search model
model = theorist.search_model_job(study_object, args.slurm_id)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)