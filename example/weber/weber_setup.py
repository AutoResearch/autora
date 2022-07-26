from types import SimpleNamespace

from experimentalist.experiment_design_synthetic_weber import (
    Experiment_Design_Synthetic_Weber,
)
from experimentalist.experiment_environment import experiment_config as exp_cfg
from experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from experimentalist.experiment_environment.variable import outputTypes as output_type
from experimentalist.experimentalist_popper import Experimentalist_Popper
from theorist.object_of_study import Object_Of_Study

# %%
# GENERAL PARAMETERS

general_parameters = SimpleNamespace(
    study_name="Weber",  # name of experiment
    study_name_sampled="Weber Sampled",
    host=exp_cfg.HOST_IP,  # ip address of experiment server
    port=exp_cfg.HOST_PORT,  # port of experiment server
)

AER_cycles = 1

# %%
# OBJECT OF STUDY

# Specify independent variables

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

# Specify dependent variable with type

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

# List dependent and independent variables

IVs = [S1, S2]  # only including subset of available variables
DVs = [diff_detected]
DVs_validation = [diff_detected_sample]

# Initialize objects of study

study_object = Object_Of_Study(
    name=general_parameters.study_name,
    independent_variables=IVs,
    dependent_variables=DVs,
)

validation_object_1 = Object_Of_Study(
    name=general_parameters.study_name_sampled,
    independent_variables=IVs,
    dependent_variables=DVs_validation,
)

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
    study_name=general_parameters.study_name,
    experiment_server_host=general_parameters.host,
    experiment_server_port=general_parameters.port,
    experiment_design=weber_design,
)

experimentalist_validation = Experimentalist_Popper(
    study_name=general_parameters.study_name_sampled,
    experiment_server_host=general_parameters.host,
    experiment_server_port=general_parameters.port,
    experiment_design=weber_design_validation,
)
