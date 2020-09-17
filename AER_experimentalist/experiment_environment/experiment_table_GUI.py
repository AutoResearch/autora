from tkinter import *
from tkinter import ttk
from AER_experimentalist.experiment_environment.experiment_toplevel_GUI import Experiment_Toplevel_GUI
from AER_experimentalist.experiment_environment.utils import *
from AER_experimentalist.experiment_environment.experiment import Experiment

class Experiment_Table_GUI(Experiment_Toplevel_GUI):

    # GUI settings
    _title = "Experiment Environment - Data"

    # Initialize GUI.
    def __init__(self, exp, update=True):

        num_rows = 2
        num_cols = 1

        Experiment_Toplevel_GUI.__init__(self, num_rows, num_cols, exp)

        self.treeview_style = ttk.Style()
        self.treeview_style.configure("mystyle.Treeview", font=(self._font_family, self._font_size))
        self.treeview_style.configure("mystyle.Treeview.Heading", font=(self._font_family, self._font_size))

        # set up data for table
        self.tree = ttk.Treeview(self._root, style="mystyle.Treeview")

        Grid.rowconfigure(self._root, 0, minsize=300)
        self.init_window(update)

    def init_window(self, update=True):

        # set up GUI
        self._root.title(self._title)
        self.tree.grid(row=0, column=0, sticky=N+S+E+W)

        # set up tree view
        self.init_tree()
        if update:
            self.update_table()

    def init_tree(self):

        # clear tree
        self.tree.delete(*self.tree.get_children())

        if self._exp is not None:
            # get variables to display
            variable_list = list()
            IV_list = self._exp.get_IV_names()
            CV_list = self._exp.get_CV_names()
            DV_list = self._exp.get_DV_names()

            if len(IV_list) > 0:
                for variable in IV_list:
                    variable_list.append(variable)

            if len(CV_list) > 0:
                for variable in CV_list:
                    variable_list.append(variable)

            if len(DV_list) > 0:
                for variable in DV_list:
                    variable_list.append(variable)

            # set columns
            self.tree["columns"] = variable_list

            # set headers
            for IV in self._exp.IVs:
                self.tree.heading(IV.get_name(), text=IV.get_variable_label(), anchor=W)

            for CV in self._exp.CVs:
                self.tree.heading(CV.get_name(), text=CV.get_variable_label(), anchor=W)

            for DV in self._exp.DVs:
                self.tree.heading(DV.get_name(), text=DV.get_variable_label(), anchor=W)

    def update_table(self):

        # clear tree
        self.tree.delete(*self.tree.get_children())

        if self._exp is not None:

            if len(self._exp.data) > 0:
                values_view = self._exp.data.values()
                value_iterator = iter(values_view)
                max_trial = len(next(value_iterator))
            else:
                max_trial = self._exp.get_current_trial() + 1

            for trial in range(max_trial):

                # generate list of values for each column
                value_list = list()

                for IV in self._exp.IVs:
                    value_list.append(str(self._exp.sequence[IV.get_name()][trial]))

                for CV in self._exp.CVs:
                    value_list.append(str(self._exp.data[CV.get_name()][trial]))

                for DV in self._exp.DVs:
                    value_list.append(str(self._exp.data[DV.get_name()][trial]))

                self.tree.insert("", END, text="Step " + str(trial), values=value_list)

