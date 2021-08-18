from AER_experimentalist.experiment_environment.experiment_server import Experiment_Server
from AER_experimentalist.experiment_environment.experiment_in_silico import Experiment_In_Silico
from AER_experimentalist.experiment_environment.participant_stroop import Participant_Stroop
from AER_experimentalist.experiment_environment.participant_weber import Participant_Weber
from AER_experimentalist.experiment_environment.participant_exp_learning import Participant_Exp_Learning
from AER_experimentalist.experiment_environment.participant_lca import Participant_LCA
import AER_experimentalist.experiment_environment.experiment_config as cfg


# define Stroop participant
# participant = Participant_Stroop() # the Cohen et al. (1990) Stroop Model
participant = Participant_Weber() # Weber contrast
# participant = Participant_Exp_Learning() # Exponential Learning Equation (Hull, 1943)
# participant = Participant_LCA() # Leaky Competitive Accumulator from Usher & McClelland (2001)

# define experiment
experiment = Experiment_In_Silico(participant=participant, main_directory=cfg.server_path)

# launch server
server = Experiment_Server(host=cfg.HOST_IP, port=cfg.HOST_PORT, exp=experiment)
server.launch()



