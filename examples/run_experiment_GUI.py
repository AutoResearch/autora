# from experiment import Experiment
#
# path = 'experiments/experiment0.exp'
# file = Experiment(path)

from aer.experiment_environment.experiment_GUI import Experiment_GUI
from tkinter import *

# init_epaper()

root = Tk()

app = Experiment_GUI(root=root)

root.mainloop()