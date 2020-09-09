from tkinter import *
from tkinter import ttk
import AER_experimentalist.experiment_environment.experiment_config as config

class Experiment_Toplevel_GUI():

    # GUI settings
    _label_color = "#DDDDDD"
    _close_color = config.stop_bgcolor
    _font_family = config.font_family
    _font_size = config.font_size_button
    _close_text = "CLOSE"

    _experiments_path = config.experiments_path
    _root = None
    _exp = None

    # Initialize GUI.
    def __init__(self, num_rows, num_cols, exp):

        self._root = Toplevel()

        self._exp = exp

        # fit grid to cell
        for row in range(num_rows):
            Grid.rowconfigure(self._root, row, weight=1)

        for col in range(num_cols):
            Grid.columnconfigure(self._root, col, weight=1)

        #Grid.columnconfigure(self._root, num_cols-1, minsize=50)

        # set styles
        self.close_button_style = ttk.Style()
        self.close_button_style.configure("Close.TButton", foreground="black", background=self._close_color,
                                            font=(self._font_family, self._font_size))

        # set up close button
        self.button_close = Button(self._root,
                                 text=self._close_text,
                                 command=self.close,
                                 bg=self._close_color,
                                 font=(self._font_family, self._font_size))

        # Close window button
        self.button_close.grid(row=num_rows-1, column=0, sticky=N + S + E + W)

        # Resize
        wpad = 3
        hpad = 67
        self._geom = '400x400+0+0'
        self._root.geometry("{0}x{1}+0+0".format(
            self._root.winfo_screenwidth() - wpad, self._root.winfo_screenheight() - hpad))
        self._root.bind('<Escape>', self.toggle_geom)

    def toggle_geom(self, event):

        geom = self._root.winfo_geometry()
        print(geom, self._geom)
        self._root.geometry(self._geom)
        self._geom = geom

    def close(self):

        self._root.destroy()
