# from experiment import Experiment
#
# path = 'experiments/experiment0.exp'
# file = Experiment(path)

from tkinter import *

from AER_experimentalist.experiment_environment.experiment_GUI import Experiment_GUI
from AER_experimentalist.experiment_environment.utils import *

# init_epaper()

root = Tk()

app = Experiment_GUI(root=root)

root.mainloop()
