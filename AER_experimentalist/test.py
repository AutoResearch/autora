# from experiment import Experiment
#
# path = 'experiments/experiment0.exp'
# file = Experiment(path)

from experiment_GUI import Experiment_GUI
from tkinter import *

root = Tk()

app = Experiment_GUI(root=root)

root.mainloop()