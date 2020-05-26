# from experiment import Experiment
#
# path = 'experiments/experiment0.exp'
# file = Experiment(path)

from experiment_server_GUI import Experiment_Server_GUI
from tkinter import *
from utils import *

# init_epaper()

root = Tk()

app = Experiment_Server_GUI(root=root)

root.mainloop()