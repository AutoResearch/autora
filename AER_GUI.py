from tkinter import *
from tkinter import ttk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import AER_config as config
from AER_utils import Plot_Types
import os
import numpy as np
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from enum import Enum

class Plot_Windows(Enum):
    THEORIST = 1
    EXPERIMENTALIST = 2

class AER_GUI(Frame):

    AER_cycles = 5

    # GUI settings
    _title = "AER"

    # general settings
    _msg_modeling_start = "# START"
    _msg_modeling_finish = "DONE"
    _default_label_status_text = "Status"
    _default_run_text = "RUN"
    _default_stop_text = "STOP"
    _label_bgcolor = "#DDDDDD"
    _up_down_bgcolor = config.up_down_bgcolor
    _run_bgcolor = "#d2ffbf"
    _stop_bgcolor = config.stop_bgcolor
    _listbox_bgcolor = "white"
    _font_family = config.font_family
    _font_size = config.font_size
    _title_font_size = config.title_font_size
    _font_size_button = config.font_size_button

    # grid parameters
    _model_plot_height = 220
    _run_button_width = 150
    _theorist_plot_width = 200
    _theorist_button_width = 30
    _experimentalist_plot_width = _theorist_plot_width

    _root = None
    _running = False
    _paused = False

    _last_meta_param_idx = 0
    _last_epoch = 0

    # plot parameters
    _reset_theorist_plot = False
    model_plot_img = None
    _plot_fontSize = 10
    _scatter_area = 50
    _scatter_color = "#FF0000"
    _plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
                    'r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--',
                    'r:', 'g:', 'b:', 'c:', 'm:', 'y:', 'k:']

    # Initialize GUI.
    def __init__(self, object_of_study, theorist, experimentalist, root=None):

        # set up window
        if root is not None:
            self._root = root

        self.object_of_study = object_of_study
        self.theorist = theorist
        self.experimentalist = experimentalist

        Frame.__init__(self, self._root)

        # define styles
        self.label_style = ttk.Style()
        self.label_style.configure("Default.TLabel", foreground="black", background=self._label_bgcolor,
                                   font=(self._font_family, self._title_font_size), anchor="center")

        self.active_label_style = ttk.Style()
        self.active_label_style.configure("Active.TLabel", foreground="red", background=self._label_bgcolor,
                                   font=(self._font_family, self._title_font_size), anchor="center")

        self.up_down_button_style = ttk.Style()
        self.up_down_button_style.configure("UpDown.TButton", foreground="black", background=self._up_down_bgcolor,
                                   font=(self._font_family, self._font_size_button))

        self.run_button_style = ttk.Style()
        self.run_button_style.configure("Run.TButton", foreground="black",
                                            background=self._run_bgcolor,
                                            font=(self._font_family, self._font_size_button))

        self.stop_button_style = ttk.Style()
        self.stop_button_style.configure("Stop.TButton", foreground="black",
                                            background=self._stop_bgcolor,
                                            font=(self._font_family, self._font_size_button))


        # configure grid
        for row in range(4):
            Grid.rowconfigure(self._root, row, weight=1)

        for col in range(5):
            Grid.columnconfigure(self._root, col, weight=1)

        # set size
        Grid.rowconfigure(self._root, 1, minsize=self._model_plot_height)
        Grid.columnconfigure(self._root, 0, minsize=self._run_button_width)
        Grid.columnconfigure(self._root, 1, minsize=self._theorist_plot_width)
        Grid.columnconfigure(self._root, 3, minsize=self._experimentalist_plot_width)

        # set up window components

        # AER control panel

        self.label_aer = ttk.Label(self._root, text='AER Status', style='Default.TLabel')

        self.listbox_status = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                          bg=self._listbox_bgcolor)

        self.button_run = ttk.Button(self._root,
                                      text=self._default_run_text,
                                      command=self.run_study,
                                      style="Run.TButton")

        self.button_stop = ttk.Button(self._root,
                                     text=self._default_stop_text,
                                     command=self.stop_study,
                                     style="Stop.TButton")



        # theorist

        self.label_theorist = ttk.Label(self._root, text='Theorist', style='Default.TLabel')

        self.model_plot_canvas = Label(self._root)

        self._fig_theorist = Figure(figsize=(1, 1), dpi=100)
        self._axis_theorist = self._fig_theorist.add_subplot(111)
        self._fig_theorist.subplots_adjust(bottom=0.2)
        self._fig_theorist.subplots_adjust(left=0.35)
        self._axis_theorist.plot([0],[0])
        self._axis_theorist.set_xlabel('Ordinate', fontsize=self._font_size)
        self._axis_theorist.set_ylabel('Epochs', fontsize=self._font_size)
        self._axis_theorist.set_title('No Data Available', fontsize=self._font_size)
        self._axis_theorist.grid()
        self._canvas_theorist = FigureCanvasTkAgg(self._fig_theorist, self._root)

        # self.model_plot_canvas = Label(self._root)
        self._theorist_canvas_width = self._theorist_plot_width + self._theorist_button_width

        self.listbox_theorist = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                          bg=self._listbox_bgcolor, exportselection=False)
        self.listbox_theorist.bind('<<ListboxSelect>>', self.update_theorist_plot)

        self.button_theorist_selection_up = ttk.Button(self._root,
                                                         text="  /\\  ",
                                                         command=self.theorist_selection_up,
                                                         style="UpDown.TButton")

        self.button_theorist_selection_down = ttk.Button(self._root,
                                                           text="  \\/  ",
                                                           command=self.theorist_selection_down,
                                                           style="UpDown.TButton")

        # experimentalist

        self.label_experimentalist = ttk.Label(self._root, text='Experimentalist', style='Default.TLabel')
        

        self.listbox_experimentalist = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                           bg=self._listbox_bgcolor, exportselection=False)
        self.listbox_experimentalist.bind('<<ListboxSelect>>', self.update_experimentalist_plot)

        self.button_experimentalist_up = ttk.Button(self._root,
                                                          text="  /\\  ",
                                                          command=self.experimentalist_selection_up,
                                                          style="UpDown.TButton")

        self.button_experimentalist_down = ttk.Button(self._root,
                                                            text="  \\/  ",
                                                            command=self.experimentalist_selection_down,
                                                            style="UpDown.TButton")

        self._fig_experimentalist = Figure(figsize=(1, 1), dpi=100)
        self._axis_experimentalist = self._fig_experimentalist.add_subplot(111)
        self._fig_experimentalist.subplots_adjust(bottom=0.2)
        self._fig_experimentalist.subplots_adjust(left=0.35)
        self._axis_experimentalist.plot([0], [0])
        self._axis_experimentalist.set_xlabel('Independent Var', fontsize=self._font_size)
        self._axis_experimentalist.set_ylabel('Dependent Var', fontsize=self._font_size)
        self._axis_experimentalist.set_title('No Data Available', fontsize=self._font_size)
        self._axis_experimentalist.grid()
        self._canvas_experimentalist = FigureCanvasTkAgg(self._fig_experimentalist, self._root)

        self.init_window()

    def init_window(self):

        # set up GUI
        self._root.title(self._title)

        # AER status
        self.label_aer.grid(row=0, column=0, sticky=N + S + E + W)
        self.listbox_status.grid(row=1, column=0, sticky=N + S + E + W)
        self.button_stop.grid(row=2, column=0, sticky=N + S + E + W)
        self.button_run.grid(row=3, column=0, sticky=N + S + E + W)


        # theorist
        self.label_theorist.grid(row=0, column=1, columnspan=2, sticky=N + S + E + W)

        self.model_plot_canvas.grid(row=1, column=1, columnspan=2, sticky=N + S + E + W)
        self._canvas_theorist.get_tk_widget().grid(row=1, column=1, columnspan=2, sticky=N + S + E + W)

        self.listbox_theorist.grid(row=2, rowspan=2, column=1, sticky=N + S + E + W)
        self.button_theorist_selection_up.grid(row=2, column=2, sticky=N + S + E + W)
        self.button_theorist_selection_down.grid(row=3, column=2, sticky=N + S + E + W)


        # experimentalist
        self.label_experimentalist.grid(row=0, column=3, columnspan=2, sticky=N + S + E + W)

        self._canvas_experimentalist.get_tk_widget().grid(row=1, column=3, columnspan=2, sticky=N + S + E + W)

        self.listbox_experimentalist.grid(row=2, rowspan=2, column=3, columnspan=2, sticky=N + S + E + W)
        self.button_experimentalist_up.grid(row=2, column=4, sticky=N + S + E + W)
        self.button_experimentalist_down.grid(row=3, column=4, sticky=N + S + E + W)


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

    def theorist_selection_up(self):

        self.move_listbox_selection(self.listbox_theorist, -1)

    def theorist_selection_down(self):

        self.move_listbox_selection(self.listbox_theorist, +1)

    def experimentalist_selection_up(self):

        self.move_listbox_selection(self.listbox_experimentalist, -1)

    def experimentalist_selection_down(self):

        self.move_listbox_selection(self.listbox_experimentalist, +1)

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

    def get_position_from_listbox_value(self, listbox, target_value):

        # find index corresponding to value of listbox item
        position = None
        for idx in range(listbox.size()):
            value = listbox.get(idx)
            if value == target_value:
                position = idx
                break

        return position

    def set_listbox_selection_to_value(self, listbox, target_value):

        position = self.get_position_from_listbox_value(listbox, target_value)

        # if the item exists, set selection of listbox to this item
        if position is not None:
            self.set_listbox_selection(listbox, position)
            return True
        else:
            return False

    def set_listbox_selection(self, listbox, position):

        listbox.selection_clear(0, END)
        listbox.select_set(position)
        listbox.activate(position)

    def update_status(self, msg):
        self.listbox_status.insert(END, msg)
        self.set_listbox_selection(self.listbox_status, self.listbox_status.size() - 1)
        self.listbox_status.yview_moveto(1)

    def update_status_theorist(self, msg):
        msg = "Theorist: " + msg
        self.update_status(msg)

    def update_status_experimentalist(self, msg):
        msg = "Experimentalist: " + msg
        self.update_status(msg)

    def activate_theorist(self):
        self.label_theorist.config(style = "Active.TLabel")
        self.label_experimentalist.config(style = "Default.TLabel")

    def activate_experimentalist(self):
        self.label_experimentalist.config(style = "Active.TLabel")
        self.label_theorist.config(style = "Default.TLabel")

    def update_run_button(self, epoch=0, num_epochs=1, meta_idx=0, num_meta_idx=0):
        if self._running is True:
            if self._paused is True:
                epoch = self._last_epoch
                num_epochs = self.theorist.model_search_epochs

            percent = np.round(epoch / num_epochs * 100)
            if percent < 10:
                str_percent = "0" + str(percent)
            else:
                str_percent = "" + str(percent)

            if self._paused is True:
                self.button_run.config(text="RESUME\n"
                                            + str(meta_idx)
                                            + " (" + str_percent + "%)"
                                            + " / "
                                            + str(num_meta_idx), command=self.resume_study)
            else:
                self.button_run.config(text="PAUSE\n"
                                            + str(meta_idx)
                                            + " (" + str_percent + "%)"
                                            + " / "
                                            + str(num_meta_idx), command=self.pause_study)
        else:
            self.button_run.config(text=self._default_run_text, command=self.run_study)

    def update_model_plot(self):
        # load image
        model_image_path = self.theorist.plot_model(self.object_of_study)

        image = mpimg.imread(model_image_path)
        self._axis_theorist.cla()
        self._axis_theorist.imshow(image)
        self._axis_theorist.get_xaxis().set_visible(False)
        self._axis_theorist.get_yaxis().set_visible(False)
        self._canvas_theorist.draw()
        self._reset_theorist_plot = True
        # needed here, otherwise canvas doesn't update
        # self._root.update()

    def get_theorist_plots(self):
        # collect performance plots
        performance_plots = self.theorist.get_performance_plots(self.object_of_study)
        # collect supplementary plots
        supplementary_plots = self.theorist.get_supplementary_plots(self.object_of_study)

        theorist_plots = {**performance_plots, **supplementary_plots}

        # add model plot
        plot_dict = dict()
        plot_dict[config.plot_key_type] = Plot_Types.MODEL
        theorist_plots["model architecture"] = plot_dict

        return theorist_plots

    def update_theorist_plot_list(self, theorist_plots):
        self.listbox_theorist.delete(0, 'end')
        keys = theorist_plots.keys()
        for key in keys:
            param_label = key
            self.listbox_theorist.insert(END, param_label)

    def get_experimentalist_plots(self):
        experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
        return experimentalist_plots

    def update_experimentalist_plot_list(self, experimentalist_plots):
        self.listbox_experimentalist.delete(0, 'end')
        keys = experimentalist_plots.keys()
        for key in keys:
            param_label = key
            self.listbox_experimentalist.insert(END, param_label)

    def update_theorist_plot(self, event):
        plots = self.get_theorist_plots()
        self.update_plot(plots=plots, plot_type=Plot_Windows.THEORIST)

    def update_experimentalist_plot(self, event):
        plots = self.get_experimentalist_plots()
        plot_name = self.update_plot(plots=plots, plot_type=Plot_Windows.EXPERIMENTALIST)
        success = self.set_listbox_selection_to_value(self.listbox_theorist, plot_name)
        if success:
            plots = self.get_theorist_plots()
            self.update_plot(plots=plots, plot_type=Plot_Windows.THEORIST)

    def update_plot(self, plots=None, plot_type=Plot_Windows.THEORIST, save=False, AER_step=1):

        if plot_type == Plot_Windows.THEORIST:

            relevant_listbox = self.listbox_theorist

            if isinstance(plots, dict) is False:
                plots = self.get_theorist_plots()

            if hasattr(self, '_axis_theorist'):
                plot_axis = self._axis_theorist

            if hasattr(self, '_canvas_theorist'):
                plot_canvas = self._canvas_theorist

        elif plot_type == Plot_Windows.EXPERIMENTALIST:
            relevant_listbox = self.listbox_experimentalist

            if isinstance(plots, dict) is False:
                if hasattr(self, 'best_model'):
                    plots = self.get_experimentalist_plots()
                else:
                    return

            if hasattr(self, '_axis_theorist'):
                plot_axis = self._axis_experimentalist

            if hasattr(self, '_canvas_theorist'):
                plot_canvas = self._canvas_experimentalist

        else:
            return


        listbox_selection = relevant_listbox.curselection()
        if len(listbox_selection) == 0:
            listbox_selection = [0]
        key = str(relevant_listbox.get(listbox_selection[0]))

        # during initial call, key is empty
        if key == '':
            return key

        if key in plots.keys():
            plot_dict = plots[key]


            if self._reset_theorist_plot:
                self._fig_theorist = Figure(figsize=(1, 1), dpi=100)
                self._axis_theorist = self._fig_theorist.add_subplot(111)
                self._fig_theorist.subplots_adjust(bottom=0.2)
                self._fig_theorist.subplots_adjust(left=0.35)
                self._axis_theorist.plot([0], [0])
                self._axis_theorist.set_xlabel('Ordinate', fontsize=self._font_size)
                self._axis_theorist.set_ylabel('Epochs', fontsize=self._font_size)
                self._axis_theorist.set_title('No Data Available', fontsize=self._font_size)
                self._axis_theorist.grid()
                self._canvas_theorist = FigureCanvasTkAgg(self._fig_theorist, self._root)
                self._canvas_theorist.get_tk_widget().grid(row=1, column=1, columnspan=2, sticky=N + S + E + W)
                self._reset_theorist_plot = False

            type = plot_dict[config.plot_key_type]
            if type == Plot_Types.LINE:

                # get relevant data
                x_data = plot_dict[config.plot_key_x_data]
                y_data = plot_dict[config.plot_key_y_data]
                x_limit = plot_dict[config.plot_key_x_limit]
                y_limit = plot_dict[config.plot_key_y_limit]
                x_label = plot_dict[config.plot_key_x_label]
                y_label = plot_dict[config.plot_key_y_label]
                legend = plot_dict[config.plot_key_legend]

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]    # remove previous lines
                plots = list()
                if isinstance(x_data, tuple) or isinstance(x_data, list):
                    for idx, (x, y, leg) in enumerate(zip(x_data, y_data, legend)):
                        plots.append(plot_axis.plot(x, y, self._plot_colors[idx], label=leg))
                else:
                    plots.append(plot_axis.plot(x_data, y_data, self._plot_colors[0], label=legend))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=self._plot_fontSize)
                plot_axis.set_ylabel(y_label, fontsize=self._plot_fontSize)

                plot_axis.legend(loc=2, fontsize="small")

                plot_canvas.draw()

            elif type == Plot_Types.MODEL:
                self.update_model_plot()
                return

            elif type == Plot_Types.IMAGE:

                # get relevant data
                image = plot_dict[config.plot_key_image]
                x_data = plot_dict[config.plot_key_x_data]
                y_data = plot_dict[config.plot_key_y_data]
                x_label = plot_dict[config.plot_key_x_label]
                y_label = plot_dict[config.plot_key_y_label]

                # generate image
                plot_axis.cla()
                plot_axis.imshow(image, interpolation='nearest', aspect='auto')
                x = x_data
                y = y_data
                plot_axis.plot(x, y, color='red')

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=self._plot_fontSize)
                plot_axis.set_ylabel(y_label, fontsize=self._plot_fontSize)

            elif type == Plot_Types.LINE_SCATTER:

                # get relevant data
                x_data = plot_dict[config.plot_key_x_data]
                y_data = plot_dict[config.plot_key_y_data]
                x_model = plot_dict[config.plot_key_x_model]
                y_model = plot_dict[config.plot_key_y_model]
                x_limit = plot_dict[config.plot_key_x_limit]
                y_limit = plot_dict[config.plot_key_y_limit]
                x_label = plot_dict[config.plot_key_x_label]
                y_label = plot_dict[config.plot_key_y_label]
                legend = plot_dict[config.plot_key_legend]

                if config.plot_key_x_conditions in plot_dict:
                    x_conditions = plot_dict[config.plot_key_x_conditions]
                else:
                    x_conditions = None

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]  # remove previous lines
                plots = list()
                # plot data
                plots.append(plot_axis.scatter(x_data, y_data, marker='.', c='k', label=legend[0]))

                # plot model prediction
                plots.append(plot_axis.plot(x_model, y_model, 'k', label=legend[1]))

                legend_idx = 1

                # plot highlighted data
                if config.plot_key_x_highlighted_data in plot_dict.keys():
                    legend_idx += 1
                    x_highlighted = plot_dict[config.plot_key_x_highlighted_data]
                    y_highlighted = plot_dict[config.plot_key_y_highlighted_data]
                    plots.append(plot_axis.scatter(x_highlighted, y_highlighted, marker='.', c='r', label=legend[legend_idx]))

                # plot conditions
                if config.plot_key_x_conditions in plot_dict.keys():
                    for idx, condition in enumerate(x_conditions):
                        if idx == 0:
                            legend_idx += 1
                        x = [condition, condition]
                        y = [y_limit[0], y_limit[1]]
                        if idx == 0:
                            plots.append(plot_axis.plot(x, y, 'b', label=legend[legend_idx]))
                        else:
                            plots.append(plot_axis.plot(x, y, 'b'))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=self._plot_fontSize)
                plot_axis.set_ylabel(y_label, fontsize=self._plot_fontSize)

                plot_axis.legend(loc=2, fontsize="small")

            elif type == Plot_Types.SURFACE_SCATTER:

                # get relevant data
                (x1_data, x2_data) = plot_dict[config.plot_key_x_data]
                y_data = plot_dict[config.plot_key_y_data]
                (x1_model, x2_model) = plot_dict[config.plot_key_x_model]
                y_model = plot_dict[config.plot_key_y_model]
                (x1_limit, x2_limit) = plot_dict[config.plot_key_x_limit]
                y_limit = plot_dict[config.plot_key_y_limit]
                (x1_label, x2_label) = plot_dict[config.plot_key_x_label]
                y_label = plot_dict[config.plot_key_y_label]
                legend = plot_dict[config.plot_key_legend]

                if config.plot_key_x_conditions in plot_dict:
                    x_conditions = plot_dict[config.plot_key_x_conditions]
                else:
                    x_conditions = None

                if config.plot_key_y_conditions in plot_dict:
                    y_conditions = plot_dict[config.plot_key_y_conditions]
                else:
                    y_conditions = None

                # generate plots
                plot_axis.cla()
                plots = list()

                # plot data
                plots.append(plot_axis.scatter(x1_data, x2_data, y_data, color = (0, 0, 0, 0), label=legend[0]))

                # plot model prediction
                plots.append(plot_axis.plot_surface(x1_model, x2_model, y_model, color=(0, 0, 0, 0.5), label=legend[1]))

                legend_idx = 1

                # plot highlighted data
                if config.plot_key_x_highlighted_data in plot_dict.keys():
                    legend_idx += 1
                    (x1_highlighted, x2_highlighted) = plot_dict[config.plot_key_x_highlighted_data]
                    y_highlighted = plot_dict[config.plot_key_y_highlighted_data]
                    plots.append(plot_axis.scatter(x1_highlighted, x2_highlighted, y_highlighted, color=(1, 0, 0, 0.5), label=legend[2]))

                # plot conditions
                for idx, x1_x2_condition in enumerate(x_conditions):
                    if idx == 0:
                        legend_idx += 1
                    x1 = [x1_x2_condition[0], x1_x2_condition[0]]
                    x2 = [x1_x2_condition[1], x1_x2_condition[1]]
                    y = [y_limit[0], y_limit[1]]
                    if idx == 0:
                        plots.append(plot_axis.plot(x1, x2, y, 'r', label=legend[legend_idx]))
                    else:
                        plots.append(plot_axis.plot(x1, x2, y, 'r'))

                # adjust axes
                plot_axis.set_xlim(x1_limit[0], x1_limit[1])
                plot_axis.set_ylim(x2_limit[0], x2_limit[1])
                plot_axis.set_zlim(y_limit[0], y_limit[1])

                # set labels
                self.mismatchPlot.set_xlabel(x1_label, fontsize=self._plot_fontSize)
                self.mismatchPlot.set_ylabel(x2_label, fontsize=self._plot_fontSize)
                self.mismatchPlot.set_zlabel(y_label, fontsize=self._plot_fontSize)

                pass

            # finalize performance plot
            plot_axis.set_title(key, fontsize=self._plot_fontSize)
            plot_canvas.draw()

            # save plot
            if save is True:
                self._root.update()
                if plot_type == Plot_Windows.THEORIST:
                    plot_filepath = os.path.join(self.theorist.results_path, 'plot_AER_step_' + str(AER_step) + '_theorist_' + key + '.png')
                    self._fig_theorist.savefig(plot_filepath)
                elif plot_type == Plot_Windows.EXPERIMENTALIST:
                    plot_filepath = os.path.join(self.theorist.results_path,
                                                 'plot_AER_step_' + str(AER_step) + '_experimentalist_' + key + '.png')
                    self._fig_experimentalist.savefig(plot_filepath)

            return key
            # plot_canvas.get_tk_widget().lift()


        else:
            raise Exception("Key '" + str(key) + "' not found in dictionary performance_plots.")

    def stop_study(self):
        self.update_status("Aborting study...")
        self._running = False
        if self._paused is True:
            self.reset_gui()

    def pause_study(self):
        # self.update_status("Pausing study...")
        # self._paused = True
        # todo: implement proper pausing
        pass

    def resume_study(self):
        self._paused = False
        self.update_status("Resuming study...")
        self.run_study(resume=True)

    def run_study(self, resume=False):

        if resume is False:
            self._running = True
            self._paused = False
            self.update_run_button()
            self._root.update()

            new_param_value = simpledialog.askstring("AER Cycles", "Please enter number of autonomous research cycles:",
                                                     parent=self._root)
            if new_param_value is not None:
                self.AER_cycles = int(new_param_value)

        # (Main) AER Loop
        for AER_cycle in range(self.AER_cycles):

            status_msg = "------------------"
            self.update_status(status_msg)
            status_msg = "AER CYCLE " + str(AER_cycle+1) + "/" + str(self.AER_cycles)
            self.update_status(status_msg)
            status_msg = "------------------"
            self.update_status(status_msg)
            self._root.update()

            # Experimentalist: collect seed data
            self.activate_experimentalist()
            if AER_cycle == 0:
                status_msg = "Collecting seed data"
                self.update_status_experimentalist(status_msg)
                self._root.update()
                seed_data = self.experimentalist.seed(self.object_of_study)
                self.object_of_study.add_data(seed_data)
            else:

                # activate experimenter
                self.activate_experimentalist()

                # Experimenter: update experimentalist plot based on best-fitting model
                status_msg = "Looking at best-fitting model..."
                self.update_status_experimentalist(status_msg)
                experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
                self.update_experimentalist_plot_list(experimentalist_plots)
                self.update_plot(plots=experimentalist_plots, plot_type=Plot_Windows.EXPERIMENTALIST)
                self._root.update()

                # Experimenter: initiating experiment search
                status_msg = "Initiating experiment search..."
                self.update_status_experimentalist(status_msg)
                self._root.update()
                self.experimentalist.init_experiment_search(self.best_model, self.object_of_study)
                experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
                self.update_experimentalist_plot_list(experimentalist_plots)
                self.update_plot(plots=experimentalist_plots, plot_type=Plot_Windows.EXPERIMENTALIST)

                # Experimenter: identifying novel experiment conditions
                status_msg = "Identifying " + str(self.experimentalist.conditions_per_experiment) \
                             + " experiment conditions..."
                self.update_status_experimentalist(status_msg)
                self._root.update()
                for condition in range(self.experimentalist.conditions_per_experiment):
                    self.experimentalist.sample_experiment_condition(self.best_model, self.object_of_study, condition)

                # plot new experiment conditions
                experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
                self.update_experimentalist_plot_list(experimentalist_plots)
                self.update_plot(plots=experimentalist_plots, plot_type=Plot_Windows.EXPERIMENTALIST)
                self._root.update()

                # write novel experiment
                status_msg = "Writing experiment..."
                self.update_status_experimentalist(status_msg)
                self._root.update()
                experiment_file_path = self.experimentalist._write_experiment(self.object_of_study, self.experimentalist._experiment_sequence)

                # collect data from experiment
                status_msg = "Commissioning experiment..."
                self.update_status_experimentalist(status_msg)
                self._root.update()
                data = self.experimentalist.commission_experiment(object_of_study=self.object_of_study, experiment_file_path=experiment_file_path)

                # add new data to object of study
                status_msg = "Adding collected data..."
                self.update_status_experimentalist(status_msg)
                self.object_of_study.add_data(data)
                experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
                self.update_experimentalist_plot_list(experimentalist_plots)
                self.update_plot(plots=experimentalist_plots, plot_type=Plot_Windows.EXPERIMENTALIST)
                self._root.update()

                # save all experimentalist plots
                experimentalist_plots = self.experimentalist.get_plots(self.best_model, self.object_of_study)
                for item in range(self.listbox_experimentalist.size()):
                    self.set_listbox_selection(self.listbox_experimentalist, item)
                    self.update_plot(plot_type=Plot_Windows.EXPERIMENTALIST, plots=experimentalist_plots, save=True, AER_step=(AER_cycle + 1))

            # Theorist: initialize meta-parameter search
            self.activate_theorist()
            status_msg = "Initializing model search"
            self.update_status_theorist(status_msg)
            self.theorist.init_meta_search(self.object_of_study)

            # Theorist: perform architecture search for different hyper-parameters
            for idx, meta_params in enumerate(self.theorist._meta_parameters):

                if resume is True:
                    if idx < self._last_meta_param_idx:
                        continue
                else:
                    status_msg = "Model search " + str(idx+1) + "/" + str(len(self.theorist._meta_parameters))
                    self.update_status_theorist(status_msg)
                    [arch_weight_decay_df, num_graph_nodes, seed] = meta_params
                    self.theorist.init_model_search(self.object_of_study)

                    # update theorist plot list
                    theorist_plots = self.get_theorist_plots()
                    self.update_theorist_plot_list(theorist_plots)

                for epoch in range(self.theorist.model_search_epochs):

                    if resume is True:
                        if epoch < self._last_epoch:
                            continue

                    # check if still running
                    if self._running is False:
                        break

                    # check if paused
                    if self._paused is True:
                        self._last_meta_param_idx = idx
                        self._last_epoch = epoch
                        self.update_run_button(meta_idx=idx+1, num_meta_idx=len(self.theorist._meta_parameters))
                        self._root.update()
                        return

                    # update run button
                    self.update_run_button(epoch=epoch+1, num_epochs=self.theorist.model_search_epochs, meta_idx=idx+1, num_meta_idx=len(self.theorist._meta_parameters))

                    self.theorist.run_model_search_epoch(epoch)

                    # update GUI
                    self._root.update()

                    # update performance plot
                    self.theorist.log_plot_data(epoch, self.object_of_study)
                    theorist_plots = self.get_theorist_plots()
                    self.update_plot(plots=theorist_plots, plot_type=Plot_Windows.THEORIST)

                if self._running is True:

                    # save all performance plots
                    theorist_plots = self.get_theorist_plots()
                    for item in range(self.listbox_theorist.size()):
                        self.set_listbox_selection(self.listbox_theorist, item)
                        self.update_plot(plot_type=Plot_Windows.THEORIST, plots=theorist_plots, save=True,
                                         AER_step=(AER_cycle + 1))

                    status_msg = "Evaluating architecture..."
                    self.update_status_theorist(status_msg)
                    self._root.update()
                    self.theorist.log_model_search(self.object_of_study)

                    # self.theorist.evaluate_model_search(self.object_of_study)

                    # Theorist: evaluate model architecture

                    # initialize meta evaluation
                    self.theorist.init_meta_evaluation(self.object_of_study)

                    # perform architecture search for different hyper-parameters
                    for eval_meta_params in self.theorist._eval_meta_parameters:

                        status_msg = "Evaluation " + str(idx + 1) + "/" + str(len(self.theorist._eval_meta_parameters))
                        self.update_status_theorist(status_msg)

                        self.theorist.init_model_evaluation(self.object_of_study)
                        # loop over epochs
                        for epoch in range(self.theorist.eval_epochs):
                            # run single epoch
                            self.theorist.run_eval_epoch(epoch, self.object_of_study)
                            # log performance (for plotting purposes)
                            self.theorist.log_plot_data(epoch, self.object_of_study)
                            theorist_plots = self.get_theorist_plots()
                            self.update_plot(plots=theorist_plots, plot_type=Plot_Windows.THEORIST)
                            self._root.update()

                        # log model evaluation
                        self.theorist.log_model_evaluation(self.object_of_study)

                    # sum up meta evaluation
                    self.theorist.log_meta_evaluation(self.object_of_study)

                    self.theorist._meta_parameters_iteration += 1

            # Theorist: determine best-fitting model
            if self._running is True:
                status_msg = "Determining best-fitting model..."
                self.update_status_theorist(status_msg)
                self._root.update()
                self.best_model = self.theorist.get_best_model(self.object_of_study)
                self.theorist.model = self.best_model




        if self._running is not True:
            # reset gui elements
            # self.reset_gui()
            pass

        self._running = False
        status_msg = "DONE"
        self.update_status(status_msg)


    def reset_gui(self):
        self.update_run_button()
        self.listbox_theorist.delete(0, 'end')
        self.listbox_experimentalist.delete(0, 'end')
        self._root.update()