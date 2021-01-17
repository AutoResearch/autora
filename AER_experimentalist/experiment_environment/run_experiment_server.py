from AER_experimentalist.experiment_environment.experiment_server import Experiment_Server
from AER_experimentalist.experiment_environment.experiment_in_silico import Experiment_In_Silico
from AER_experimentalist.experiment_environment.participant_stroop import Participant_Stroop
import AER_experimentalist.experiment_environment.experiment_config as cfg

# define participant
participant = Participant_Stroop() # the Cohen et al. (1990) Stroop Model

# define experiment
experiment = Experiment_In_Silico(participant=participant, main_directory=cfg.server_path)

# launch server
server = Experiment_Server(host=cfg.HOST_IP, port=cfg.HOST_PORT, exp=experiment)
server.launch()

# exp_path = 'server data/experiments/experiment_0.exp'
# experiment.load_experiment(exp_path)
# experiment.run_experiment()
# data_file_path = cfg.server_path + 'stroop_data.csv'
# experiment.data_to_csv(data_file_path)