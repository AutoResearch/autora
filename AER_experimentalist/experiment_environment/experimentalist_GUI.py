from tkinter import *
from tkinter import ttk
from AER_experimentalist.experiment_environment.utils import *
from AER_experimentalist.experiment_environment.experiment import Experiment
from AER_experimentalist.experiment_environment.OLED_output import OLED_Output
from AER_experimentalist.experiment_environment.experiment_table_GUI import Experiment_Table_GUI
from AER_experimentalist.experiment_environment.experiment_plot_GUI import Experiment_Plot_GUI
import AER_experimentalist.experiment_environment.experiment_config as config
import os
import types

class Experimentalist_GUI(Frame):

    # GUI settings
    _title = "Experiment Environment"
    _experiments_path = config.server_path + config.experiments_path
    _sequences_path = config.server_path + config.sequences_path
    _data_path = config.server_path + config.data_path

    _msg_experiment_start = "# START"
    _msg_experiment_finish = "DONE"
    _default_label_status_text = "Status"
    _label_bgcolor = "#DDDDDD"
    _up_down_bgcolor = config.up_down_bgcolor
    _run_exp_bgcolor = "#d2ffbf"
    _stop_exp_bgcolor = config.stop_bgcolor
    _visualize_exp_bgcolor = config.visualize_bgcolor
    _IV_listbox_bgcolor = "#ffffff"
    _DV_listbox_bgcolor = "#ffffff"
    _IV_fgcolor = "#000000"
    _DV_fgcolor = "#000000"
    _listbox_bgcolor = "white"
    _font_family = config.font_family
    _font_size = config.font_size
    _font_size_button = config.font_size_button

    _bulk_output = False
    _use_OLED = False

    _root = None
    _abort = False

    experiment_server = None

    # Initialize GUI.
    def __init__(self, root=None, path=None):

        # set up window
        if root is not None:
            self._root = root

        Frame.__init__(self, self._root)

        # define styles
        self.label_style = ttk.Style()
        self.label_style.configure("Default.TLabel", foreground="black", background=self._label_bgcolor,
                                   font=(self._font_family, self._font_size), anchor="center")

        self.IV_label_style = ttk.Style()
        self.IV_label_style.configure("IV.TLabel", foreground=self._IV_fgcolor, background="white",
                                   font=(self._font_family, self._font_size_button), anchor="center")

        self.DV_label_style = ttk.Style()
        self.DV_label_style.configure("DV.TLabel", foreground=self._DV_fgcolor, background="white",
                                      font=(self._font_family, self._font_size_button), anchor="center")

        self.up_down_button_style = ttk.Style()
        self.up_down_button_style.configure("UpDown.TButton", foreground="black", background=self._up_down_bgcolor,
                                   font=(self._font_family, self._font_size_button))

        self.run_exp_button_style = ttk.Style()
        self.run_exp_button_style.configure("Run.TButton", foreground="black",
                                            background=self._run_exp_bgcolor,
                                            font=(self._font_family, self._font_size_button))

        self.stop_exp_button_style = ttk.Style()
        self.stop_exp_button_style.configure("Stop.TButton", foreground="black",
                                            background=self._stop_exp_bgcolor,
                                            font=(self._font_family, self._font_size_button))

        self.visualize_exp_button_style = ttk.Style()
        self.visualize_exp_button_style.configure("Viz.TButton", foreground="black",
                                             background=self._visualize_exp_bgcolor,
                                             font=(self._font_family, self._font_size_button))


        # fit grid to cell
        for row in range(3):
            Grid.rowconfigure(self._root, row, weight=1)

        for col in range(4):
            Grid.columnconfigure(self._root, col, weight=1)

        # set size
        Grid.rowconfigure(self._root, 3, minsize=80)
        Grid.rowconfigure(self._root, 0, minsize=80)
        Grid.columnconfigure(self._root, 6, minsize=200)

        # set up window components
        self.label_status = ttk.Label(self._root, text=self._default_label_status_text, style="Default.TLabel")
        self.listbox_status = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size))

        self.listbox_IVs = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                   bg=self._IV_listbox_bgcolor)
        self.listbox_DVs = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                   bg=self._DV_listbox_bgcolor)
        self.listbox_output = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size))

        self.button_IV_selection_up = ttk.Button(self._root,
                                             text="  /\\  ",
                                             command=self.IV_selection_up,
                                             style="UpDown.TButton")

        self.button_IV_selection_down = ttk.Button(self._root,
                                               text="  \\/  ",
                                               command=self.IV_selection_down,
                                               style="UpDown.TButton")

        self.button_DV_selection_up = ttk.Button(self._root,
                                             text="  /\\  ",
                                             command=self.DV_selection_up,
                                             style="UpDown.TButton")

        self.button_DV_selection_down = ttk.Button(self._root,
                                               text="  \\/  ",
                                               command=self.DV_selection_down,
                                               style="UpDown.TButton")

        self.button_status_selection_up = ttk.Button(self._root,
                                                     text="  /\\  ",
                                                     style="UpDown.TButton")

        self.button_status_selection_down = ttk.Button(self._root,
                                                       text="  \\/  ",
                                                       style="UpDown.TButton")

        self.button_plot = ttk.Button(self._root,
                                     text="PLOT ",
                                     command=self.plot_experiment,
                                     style="Viz.TButton")

        self.button_table = ttk.Button(self._root,
                                     text="SHOW TABLE ",
                                     command=self.table_experiment,
                                     style="Viz.TButton")

        # bind events
        self.listbox_IVs.bind('<<ListboxSelect>>', self.select_IV)
        self.listbox_DVs.bind('<<ListboxSelect>>', self.select_DV)

        # set up experiment path
        if path is not None:
            self._experiments_path = path

        # set up GUI variables
        self._plot_GUI = None
        self._table_GUI = None
        self._exp = None
        self._exp_name = None
        self._IV_name = None
        self._DV_name = None

        # set up OLED display
        if self._use_OLED:
            self._OLED = OLED_Output()

        self.init_window()

    def init_window(self):

        # set up GUI
        self._root.title(self._title)

        # status panel
        self.label_status.grid(row=0, column=0, sticky=N + S + E + W)

        self.listbox_status.grid(rowspan=2, column=0, sticky=N + S + E + W)

        self.button_status_selection_up.grid(row=1, column=1, sticky=N + S + E + W)
        self.button_status_selection_down.grid(row=2, column=1, sticky=N + S + E + W)

        # IVs
        label = ttk.Label(self._root, text="IVs", style="IV.TLabel")
        label.grid(row=0, column=2, sticky=N+S+E+W)
        self.listbox_IVs.grid(rowspan=2, row=1, column=2, sticky=N+S+E+W)
        self.button_IV_selection_up.grid(row=1, column=3, sticky=N + S + E + W)
        self.button_IV_selection_down.grid(row=2, column=3, sticky=N + S + E + W)

        # DVs
        label = ttk.Label(self._root, text="DVs", style="DV.TLabel")
        label.grid(row=0, column=4, sticky=N+S+E+W)
        self.listbox_DVs.grid(rowspan=2, row=1, column=4, sticky=N+S+E+W)
        self.button_DV_selection_up.grid(row=1, column=5, sticky=N + S + E + W)
        self.button_DV_selection_down.grid(row=2, column=5, sticky=N + S + E + W)

        # Experiment output
        self.listbox_output.grid(rowspan=2, row=1, column=6, sticky=N+S+E+W)

        # Plot experiment
        self.button_plot.grid(row=3, column=2, columnspan=4, sticky=N + S + E + W)

        # Table experiment
        self.button_table.grid(row=3, column=6, sticky=N + S + E + W)

        # resize
        hpad = 67
        wpad = 3
        self._geom = '800x400+0+0'
        self._root.geometry("{0}x{1}+0+0".format(
            self._root.winfo_screenwidth() - wpad, self._root.winfo_screenheight() - hpad))
        self._root.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):
        geom = self._root.winfo_geometry()
        print(geom, self._geom)
        self._root.geometry(self._geom)
        self._geom = geom

    def DV_selection_up(self):

        self.move_listbox_selection(self.listbox_DVs, -1)
        self.set_DV_name()

    def DV_selection_down(self):

        self.move_listbox_selection(self.listbox_DVs, +1)
        self.set_DV_name()

    def IV_selection_up(self):

        self.move_listbox_selection(self.listbox_IVs, -1)
        self.set_IV_name()

    def IV_selection_down(self):

        self.move_listbox_selection(self.listbox_IVs, +1)
        self.set_IV_name()

    def set_DV_name(self, name=None):

        if name is None:
            if len(self.listbox_DVs.curselection()) > 0:
                index = int(self.listbox_DVs.curselection()[0])
                value = self.listbox_DVs.get(index)
                for DV in self._exp.DVs:
                    if DV.get_variable_label() == value:
                        self._DV_name = DV.get_name()
        else:
            self._DV_name = name

        self.update_plot_button()

    def set_IV_name(self, name=None):

        if name is None:
            if len(self.listbox_IVs.curselection()) > 0:
                index = int(self.listbox_IVs.curselection()[0])
                value = self.listbox_IVs.get(index)
                for IV in self._exp.IVs:
                    if IV.get_variable_label() == value:
                        self._IV_name = IV.get_name()
        else:
            self._IV_name = name

        self.update_plot_button()

    def move_listbox_selection(self, listbox, movement):

        selection = listbox.curselection()

        if len(selection) > 0:
            current_value = selection[0]
            max_value = listbox.size()
            new_value = max(min(current_value + movement, max_value-1), 0)

            listbox.selection_clear(0, END)
            listbox.select_set(new_value)
            listbox.activate(new_value)

        elif listbox.size() > 0:
            listbox.select_set(0)
            listbox.activate(0)

    def set_listbox_selection(self, listbox, position):

        listbox.selection_clear(0, END)
        listbox.select_set(position)
        listbox.activate(position)

    def update_plot_button(self):

        if self._IV_name is None and self._DV_name is None:
            self.button_plot.config(text="PLOT")
            return

        if self._IV_name is not None:
            IV = self._exp.get_IV(self._IV_name).get_variable_label()
        else:
            IV = ""

        if self._DV_name is not None:
            DV = self._exp.get_DV_CV(self._DV_name).get_variable_label()
        else:
            DV = ""

        self.button_plot.config(text="PLOT: " + DV + " ~ " + IV)

    def select_IV(self, evt):
        self.set_IV_name()

    def select_DV(self, evt):
        self.set_DV_name()


    def load_experiment(self, experiment_name=None):

        if experiment_name is None:
            experiment_name = self._exp_name

        if experiment_name is not None:
            self.label_status.config(text=experiment_name)

            # load experiment
            file_path = os.path.join(self._experiments_path, experiment_name)
            self._exp = Experiment(file_path, main_directory=config.server_path)
            self._exp_name = experiment_name

            # read experiment variables
            IVs = self._exp.get_IV_labels()
            DVs = self._exp.get_DV_labels()
            CVs = self._exp.get_CV_labels()

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

        else:
            self.label_status.config(text=self._default_label_status_text)

        self._IV_name = None
        self._DV_name = None
        self.update_plot_button()

    def plot_experiment(self):
        if self._exp is not None:
            self._plot_GUI = Experiment_Plot_GUI(exp=self._exp, IV=self._exp.get_IV(self._IV_name),
                                             DV=self._exp.get_DV_CV(self._DV_name))

        else:
            self._plot_GUI = Experiment_Plot_GUI(exp=None)

    # show window for data table
    def table_experiment(self):
        self._table_GUI = Experiment_Table_GUI(exp=self._exp)


    def update_output(self, messages):

        if isinstance(messages, str):
            self.listbox_output.insert(END, messages)
        else:
            for msg in messages:
                self.listbox_output.insert(END, msg)

        self.set_listbox_selection(self.listbox_output, self.listbox_output.size()-1)
        self.listbox_output.yview_moveto(1)

    def update_OLED(self, messages):
        if self._use_OLED:
            self._OLED.clear_display()
            self._OLED.append_and_show_message(messages)

    def clear_output(self):
        self.listbox_output.delete(0, 'end')

    def clear_OLED(self):
        if self._use_OLED:
            self._OLED.clear_messages()
            self._OLED.clear_display()

    def init_run(self):
        pass

    def stop_experiment(self):
        self._abort = True

    def run_experiment(self, plot=False):

        if self._exp is None:
            return

        self._abort = False

        # clear experiment output
        self.clear_output()
        self.clear_OLED()

        # clear windows
        if self._table_GUI is not None:
            self._table_GUI.close()
            self._table_GUI = None

        if self._plot_GUI is not None:
            self._plot_GUI.close()
            self._plot_GUI = None

        if plot is True:
            self.table_experiment()

        # initialize experiment
        self.update_output(self._msg_experiment_start)
        msg = "Running " + self._exp_name
        self.update_output(msg)
        self.update_OLED(msg)

        self._exp.init_experiment()
        self.init_STOP_button()

        if self._bulk_output is True:
            bulk_message = list()

        # run experiment
        for trial in self._exp.actual_trials:

            if self._abort:
                self.load_experiment()
                break

            self._exp.set_current_trial(trial)

            # update displays with IVs
            msg = trial_to_list(trial=trial, IVList = self._exp.current_IVs_to_list())

            if self._bulk_output is False:
                self.update_output(msg)
                self.clear_OLED()
                self.update_OLED(msg)
                self.update_STOP_button((trial+1)/len(self._exp.actual_trials)*100)
                self._root.update()

            # run trial
            self._exp.run_trial()

            # update displays with DVs
            if self._bulk_output is False:
                msg = trial_to_list(DVList=self._exp.current_DVs_to_list())
            else:
                msg = trial_to_list(trial=trial, IVList=self._exp.current_IVs_to_list(), DVList=self._exp.current_DVs_to_list())
                for message in msg:
                    bulk_message.append(message)

            if self._bulk_output is False or self._exp.is_end_of_trial() is True:

                if self._bulk_output is False:
                    self.update_output(msg)
                else:
                    self.update_output(bulk_message)
                    bulk_message = list()
                self.update_OLED(msg)
                if self._table_GUI is not None:
                    self._table_GUI.update_table()
                    pass
                if self._plot_GUI is not None:
                    self._plot_GUI.update_plot()
                self._root.update()

        # display end of experiment
        if self._abort is True:
            msg = "ABORTED"
        else:
            self._exp.data_to_csv(config.server_path + self._exp._data_path) # write experiment data to csv
            msg = self._msg_experiment_finish

        if plot is True and self._table_GUI is not None:
            self._table_GUI.close()
            self._table_GUI = None

        # if self.experiment_server is not None:
        #     self.experiment_server._wrap_up_experiment(self._exp._data_path)

        self.update_output(msg)
        self.clear_OLED()
        self.update_OLED(msg)
        self.init_run()
        self._root.update()
        self._exp.clean_up()

        return self._exp._data_path

    # def open_automated_plot(self):
    #
    #     if self._IV_name is None:
    #
    #         if self.listbox_IVs.size() > 0:
    #             self.listbox_IVs.select_set(0)
    #             self.listbox_IVs.activate(0)
    #             self.set_IV_name()
    #
    #     if self._DV_name is None:
    #
    #         if self.listbox_DVs.size() > 0:
    #             self.listbox_DVs.select_set(0)
    #             self.listbox_DVs.activate(0)
    #             self.set_DV_name()
    #
    #     self.update_plot_button()
    #     self.plot_experiment()
