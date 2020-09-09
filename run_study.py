from AER_experimentalist.experiment_environment.variable import Variable
from AER_experimentalist.experimentalist import Experimantalist
from AER_theorist.object_of_study import Object_Of_Study
import AER_experimentalist.experiment_environment.experiment_config as exp_env_cfg

# GENERAL PARAMETERS

study_name = "Simple Voltage"   # name of experiment
host = "192.168.188.27" # exp_env_cfg.HOST_IP      # ip address of experiment server
port = 47777 # exp_env_cfg.HOST_PORT    # port of experiment server

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
study_object = Object_Of_Study(independent_variables=IVs,
                               dependent_variables=DVs)

# EXPERIMENTALIST

# initialize experimentalist
experimentalist = Experimantalist(study_name=study_name,
                                  object_of_study=study_object,
                                  experiment_server_host=host,
                                  experiment_server_port=port)
# seed object of study with data
experimentalist.seed()

# THEORIST

# AUTONOMOUS EMPIRICAL RESEARCH

for cycle in range(AER_cycles):
    pass

