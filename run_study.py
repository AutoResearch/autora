from AER_experimentalist.experiment_environment.variable import Variable
from AER_experimentalist.experimentalist import Experimantalist
from AER_theorist.object_of_study import Object_Of_Study
from AER_theorist.theorist import *

# GENERAL PARAMETERS

study_name = "Simple Voltage"   # name of experiment
host = "192.168.188.27" # exp_env_cfg.HOST_IP      # ip address of experiment server
port = 47778 # exp_env_cfg.HOST_PORT    # port of experiment server

AER_cycles = 1

# OBJECT OF STUDY

# specify independent variable
source_voltage = Variable(name='source_voltage',
                          value_range=(0, 4000),
                          units="mV",
                          variable_label='Source Voltage')

# specify dependent variable
target_voltage = Variable(name='voltage0',
                          units="mV",
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
experimentalist = Experimantalist(study_name=study_name,
                                  experiment_server_host=host,
                                  experiment_server_port=port)

# THEORIST
theorist = Theorist(object_of_study=study_object, architecture_search_strategy=architecture_search_strategy.DARTS)


# AUTONOMOUS EMPIRICAL RESEARCH

# seed object of study with data
seed_data = experimentalist.seed(study_object)
study_object.add_data(seed_data)

for cycle in range(AER_cycles):

    # generate computational model to explain object of study
    model = theorist.run_model_search(study_object)

    # generate experiment based on the generated computational model and object of study
    experiment_file_path = experimentalist.sample_experiment(model, study_object)

    # collect data from experiment
    data = experimentalist.commission_experiment(experiment_file_path)

    # add new data to object of study
    study_object.add_data(data)

# TODO: for AER_class, make sure to log the cycle state, so that the cycle can be reinitiated at the point it was last interrupted