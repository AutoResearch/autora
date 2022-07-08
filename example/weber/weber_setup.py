# GENERAL PARAMETERS
from types import SimpleNamespace

from experimentalist.experiment_environment import experiment_config as exp_cfg

params = SimpleNamespace(
    study_name="Weber",  # name of experiment
    study_name_sampled="Weber Sampled",
    host=exp_cfg.HOST_IP,  # ip address of experiment server
    port=exp_cfg.HOST_PORT,  # port of experiment server
)
