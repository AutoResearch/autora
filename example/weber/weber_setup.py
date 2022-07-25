from types import SimpleNamespace

from experimentalist.experiment_environment import experiment_config as exp_cfg
from experimentalist.experiment_environment.DV_in_silico import DV_In_Silico as DV
from experimentalist.experiment_environment.IV_in_silico import IV_In_Silico as IV
from experimentalist.experiment_environment.variable import outputTypes as output_type
from theorist.object_of_study import Object_Of_Study

# %%
# GENERAL PARAMETERS

general_params = SimpleNamespace(
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
    name=general_params.study_name, independent_variables=IVs, dependent_variables=DVs
)

validation_object_1 = Object_Of_Study(
    name=general_params.study_name_sampled,
    independent_variables=IVs,
    dependent_variables=DVs_validation,
)
