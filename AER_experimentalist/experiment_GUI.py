from tkinter import *
from utils import *
from experiment import Experiment
from OLED_output import OLED_Output
import os

class Experiment_GUI(Frame):

    # GUI settings
    _title = "Experiment Environment"
    _experiments_path = "experiments/"

    _msg_experiment_start = "START"
    _msg_experiment_finish = "DONE"

    _root = None
    _use_OLED = False
    _abort = False

    # Initialize GUI.
    def __init__(self, root=None, path=None):

        # set up window
        if root is not None:
            self._root = root

        Frame.__init__(self, self._root)

        # fit grid to cell
        for row in range(3):
            Grid.rowconfigure(self._root, row, weight=1)

        for row in range(4):
            Grid.rowconfigure(self._root, row, weight=1)

        # set up window components
        self.listbox_experiments = Listbox(self._root, selectmode=SINGLE)
        self.listbox_IVs = Listbox(self._root, selectmode=SINGLE)
        self.listbox_DVs = Listbox(self._root, selectmode=SINGLE)
        self.listbox_output = Listbox(self._root, selectmode=SINGLE)
        self.button_refresh = Button(self._root, text="LOAD", fg="blue", command=self.load_experiment, bg="#fbfc9f")
        self.button_selection_up = Button(self._root, text="  /\\  ", command=self.exp_selection_up, bg="#6b87b5")
        self.button_selection_down = Button(self._root, text="  \\/  ", command=self.exp_selection_down, bg="#6b87b5")
        self.button_run = Button(self._root, text="RUN EXPERIMENT", command=self.run_experiment, bg="#d2ffbf")

        # bind events
        self.listbox_experiments.bind('<<ListboxSelect>>', self.select_experiment)

        # set up experiment path
        if path is not None:
            self._experiments_path = path

        # set up OLED display
        if self._use_OLED:
            self._OLED = OLED_Output()

        self.init_window()

    def init_window(self):

        # set up GUI
        self._root.title(self._title)

        # experiment selection panel
        label = Label(self._root, text="Experiment File")
        label.grid(row=0, column=0, sticky=N+S+E+W)

        experiment_files = get_experiment_files(self._experiments_path)

        for file in experiment_files:
            self.listbox_experiments.insert(END, file)
        self.listbox_experiments.grid(rowspan=2, column=0, sticky=N+S+E+W)
        if self.listbox_experiments.size() > 0:
            self.listbox_experiments.select_set(0)
            self.listbox_experiments.activate(0)

        self.button_refresh.grid(row=0, column=1, sticky=N+S+E+W)
        self.button_selection_up.grid(row=1, column=1, sticky=N+S+E+W)
        self.button_selection_down.grid(row=2, column=1, sticky=N+S+E+W)

        # IVs
        label = Label(self._root, text="IVs")
        label.grid(row=0, column=2, sticky=N+S+E+W)
        self.listbox_IVs.grid(rowspan=2, row=1, column=2, sticky=N+S+E+W)

        # DVs
        label = Label(self._root, text="DVs")
        label.grid(row=0, column=3, sticky=N+S+E+W)
        self.listbox_DVs.grid(rowspan=2, row=1, column=3, sticky=N+S+E+W)

        # Experiment Output
        self.button_run.grid(row=0, column=4, sticky=N+S+E+W)
        self.listbox_output.grid(rowspan=2, row=1, column=4, sticky=N+S+E+W)

        # resize
        pad = 3
        self._geom = '800x200+0+0'
        self._root.geometry("{0}x{1}+0+0".format(
            self._root.winfo_screenwidth() - pad, self._root.winfo_screenheight() - pad))
        self._root.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):
        geom = self._root.winfo_geometry()
        print(geom, self._geom)
        self._root.geometry(self._geom)
        self._geom = geom

    def exp_selection_up(self):

        selection = self.listbox_experiments.curselection()

        if len(selection) > 0:
            current_value = selection[0]
            new_value = max(current_value - 1, 0)

            self.listbox_experiments.selection_clear(0, END)
            self.listbox_experiments.select_set(new_value)
            self.listbox_experiments.activate(new_value)

            self.load_experiment(self.listbox_experiments.get(new_value))

    def exp_selection_down(self):

        selection = self.listbox_experiments.curselection()

        if len(selection) > 0:
            current_value = selection[0]
            max_value = self.listbox_experiments.size()
            new_value = min(current_value + 1, max_value-1)

            self.listbox_experiments.selection_clear(0, END)
            self.listbox_experiments.select_set(new_value)
            self.listbox_experiments.activate(new_value)

            self.load_experiment(self.listbox_experiments.get(new_value))

    def update_experiments(self):

        # clear list box
        self.listbox_experiments.delete(0, END)

        # find all experiment files
        experiment_files = get_experiment_files(self._experiments_path)

        # add experiment files to listbox
        for file in experiment_files:
            self.listbox_experiments.insert(END, file)
        self.listbox_experiments.grid(rowspan=2, column=0)

    def select_experiment(self, evt):

        # fetch selection
        w = evt.widget

        if len(w.curselection()) > 0:

            index = int(w.curselection()[0])
            value = w.get(index)
            self._exp_name = value

        self.update_experiments()


    def load_experiment(self, experiment_name=None):

        if experiment_name is None:
            experiment_name = self._exp_name

        # load experiment
        file_path = os.path.join(self._experiments_path, experiment_name)
        self._exp = Experiment(file_path)
        self._exp_name = experiment_name

        # read experiment variables
        IVs = self._exp.get_IV_names()
        DVs = self._exp.get_DV_names()
        CVs = self._exp.get_CV_names()

        # clear list box
        self.listbox_IVs.delete(0, 'end')
        self.listbox_DVs.delete(0, 'end')

        # add variables to listbox
        if len(IVs) > 0:
            for IV in IVs:
                self.listbox_IVs.insert(END, IV)

        if len(DVs) > 0:
            for DV in DVs:
                self.listbox_DVs.insert(END, DV)

        if len(CVs) > 0:
            for CV in CVs:
                self.listbox_DVs.insert(END, CV + " (covariate)")

    def update_output(self, messages):

        for msg in messages:
            self.listbox_Output.insert(END, msg)

    def update_OLED(self, messages):
        if self._use_OLED:
            self._OLED.append_and_show_message(messages)

    def clear_output(self):
        self.listbox_output.delete(0, 'end')

    def clear_OLED(self):
        if self._use_OLED:
            self._OLED.clear_messages()
            self._OLED.clear_display()

    def init_RUN_button(self):
        self.button_run.configure(text="RUN EXPERIMENT", command=self.run_experiment, bg="green")

    def init_STOP_button(self):
        self.button_run.configure(text="STOP", command=self.stop_experiment, bg="red")

    def update_STOP_button(self, progress):
        self.button_run.configure(text="STOP" + "(" + str(progress) + "%)")

    def stop_experiment(self):
        self._abort = True

    def run_experiment(self):

        if self._exp is None:
            return

        self._abort = False

        # clear experiment output
        self.clear_output()
        self.clear_OLED()

        # initialize experiment
        self.update_output(self._msg_experiment_start)
        msg = "RUNNING " + self._exp_name
        self.update_output(msg)
        self.update_OLED(msg)

        self._exp.init_experiment()

        # run experiment
        for trial in self._exp.actual_trials:

            if self._abort:
                msg = "ABORTED"
                self.update_output(msg)
                self.update_OLED(msg)
                self.init_RUN_button()
                self.load_experiment()
                break

            self._exp.set_current_trial(trial)

            # update displays with IVs
            msg = trial_to_list(trial, IVList = self._exp.current_IVs_to_list())
            self.update_output(msg)
            self.clear_OLED()
            self.update_OLED(msg)
            self.update_STOP_button(trial/self._exp.actual_trials)

            # run trial
            self._exp.run_trial()

            # update displays with DVs
            msg = trial_to_list(DVList=self._exp.current_DVs_to_list())
            self.update_output(msg)
            self.update_OLED(msg)

            # initiate ITI
            self._exp.ITI()

        # display end of experiment
        self.update_output(self._msg_experiment_stop)
        self.clear_OLED()
        self.update_OLED(self._msg_experiment_stop)