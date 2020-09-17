from tkinter import *
from tkinter import ttk
from AER_experimentalist.experiment_environment.experiment_toplevel_GUI import Experiment_Toplevel_GUI

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from experiment import Experiment

class Experiment_Plot_GUI(Experiment_Toplevel_GUI):

    # GUI settings
    _title = "Experiment Environment - Plot"

    _scatter_area = 50
    _scatter_color = "#FF0000"

    # Initialize GUI.
    def __init__(self, exp, IV=None, DV=None):

        num_rows = 2
        num_cols = 1

        self.IV = IV
        self.DV = DV

        Experiment_Toplevel_GUI.__init__(self, num_rows, num_cols, exp)
        Grid.rowconfigure(self._root, 0, minsize=300)
        self.init_window()

    def init_window(self):

        # set up GUI
        self._root.title(self._title)

        self._fig = Figure(figsize=(3, 3), dpi=100)
        self._axis = self._fig.add_subplot(111)
        self._axis.scatter([0], [0], s=self._scatter_area, facecolors='none', edgecolors=self._scatter_color)
        self._axis.set_xlabel('Independent Variable', fontsize=self._font_size)
        self._axis.set_ylabel('Dependent Variable', fontsize=self._font_size)
        self._axis.set_title('No Data Available', fontsize=self._font_size)
        self._axis.grid()

        self._canvas = FigureCanvasTkAgg(self._fig, self._root)
        # canvas.show()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky=N+S+E+W)

        # update plot
        self.update_plot()


    def update_plot(self):

        if self._exp is not None and self.IV is not None and self.DV is not None:

            current_trial = len(self._exp.data[self.DV.get_name()])

            # get all data points until current trial
            x = self._exp.sequence[self.IV.get_name()][0:current_trial]
            y = self._exp.data[self.DV.get_name()][0:current_trial]

            self._axis.clear()
            self._axis.scatter(x, y, s=self._scatter_area, facecolors='none', edgecolors=self._scatter_color)

            self._axis.set_xlabel(self.IV.get_variable_label() + ' (' + self.IV.get_units() + ')', fontsize=self._font_size)
            self._axis.set_ylabel(self.DV.get_variable_label() + ' (' + self.DV.get_units() + ')', fontsize=self._font_size)
            self._axis.set_title('Live Plot')
            self._axis.grid()

            self._canvas.draw()
