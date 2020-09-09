from tkinter import *
from tkinter import ttk
from AER_experimentalist.experiment_environment.utils import *
from AER_experimentalist.experiment_environment.experimentalist_GUI import Experimentalist_GUI
import AER_experimentalist.experiment_environment.experiment_config as config

class Experiment_GUI(Experimentalist_GUI):

    _load_exp_bgcolor = "#fbfc9f"
    _default_label_status_text = "Experiment File"

    _experiments_path = config.server_path + config.experiments_path
    _sequences_path = config.server_path + config.sequences_path
    _data_path = config.server_path + config.data_path

    # Initialize GUI.
    def __init__(self, root=None, path=None):

        super().__init__(root=root, path=path)

        self.load_exp_button_style = ttk.Style()
        self.load_exp_button_style.configure("Load.TButton", foreground="black",
                                            background=self._load_exp_bgcolor,
                                            font=(self._font_family, self._font_size_button))


        # set up window components

        self.button_load = ttk.Button(self._root,
                                     text="LOAD",
                                     command=self.load_experiment,
                                     style="Load.TButton")

        self.button_status_selection_up = ttk.Button(self._root,
                                              text="  /\\  ",
                                              command=self.exp_selection_up,
                                              style="UpDown.TButton")

        self.button_status_selection_down = ttk.Button(self._root,
                                                text="  \\/  ",
                                                command=self.exp_selection_down,
                                                style="UpDown.TButton")


        self.button_run = ttk.Button(self._root,
                                 text="RUN EXPERIMENT",
                                 command=self.run_experiment,
                                 style="Run.TButton")

        # bind events
        self.listbox_status.bind('<<ListboxSelect>>', self.select_experiment)

        # experiment selection panel

        experiment_files = get_experiment_files(self._experiments_path)

        for file in experiment_files:
            self.listbox_status.insert(END, file)

        if self.listbox_status.size() > 0:
            self.listbox_status.select_set(0)
            self.listbox_status.activate(0)
            self.set_experiment_name()

        self.button_load.grid(row=3, column=0, columnspan=2, sticky=N+S+E+W)

        # Experiment output
        self.button_run.grid(row=0, column=6, sticky=N+S+E+W)

    def exp_selection_up(self):

        self.move_listbox_selection(self.listbox_status, -1)
        self.set_experiment_name()

    def exp_selection_down(self):

        self.move_listbox_selection(self.listbox_status, +1)
        self.set_experiment_name()

    def set_experiment_name(self, name=None):

        if name is None:
            if len(self.listbox_status.curselection()) > 0:
                index = int(self.listbox_status.curselection()[0])
                value = self.listbox_status.get(index)
                self._exp_name = value
        else:
            self._exp_name = name

    def update_experiments(self):

        # clear list box
        self.listbox_status.delete(0, END)

        # find all experiment files
        experiment_files = get_experiment_files(self._experiments_path)

        # add experiment files to listbox
        for file in experiment_files:
            self.listbox_status.insert(END, file)
        self.listbox_status.grid(rowspan=2, column=0)

    def select_experiment(self, evt):
        self.set_experiment_name()

    def init_run(self):
        self.button_run.configure(text="RUN EXPERIMENT", command=self.run_experiment, style="Run.TButton")

    def init_STOP_button(self):
        self.button_run.configure(text="STOP", command=self.stop_experiment, style="Stop.TButton")

    def update_STOP_button(self, progress):
        self.button_run.configure(text="STOP" + "(" + str(round(progress)) + "%)")

