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

from enum import Enum

class Plot_Windows(Enum):
    PERFORMANCE = 1
    SUPPLEMENTARY = 2

class Theorist_GUI(Frame):

    # GUI settings
    _title = "Theorist"

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
    _font_size_button = config.font_size_button

    # grid parameters
    _model_plot_height = 250
    _run_button_width = 100
    _parameter_listbox_width = 150
    _parameter_button_width = 30

    _root = None
    _running = False
    _paused = False

    _last_meta_param_idx = 0
    _last_epoch = 0

    # plot parameters
    model_plot_img = None
    _2d_plot = True
    _plot_fontSize = 10
    _scatter_area = 50
    _scatter_color = "#FF0000"
    _plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
                    'r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--',
                    'r:', 'g:', 'b:', 'c:', 'm:', 'y:', 'k:']

    # Initialize GUI.
    def __init__(self, object_of_study, theorist, root=None):

        # set up window
        if root is not None:
            self._root = root

        self.theorist = theorist
        self.object_of_study = object_of_study

        Frame.__init__(self, self._root)

        # define styles
        self.label_style = ttk.Style()
        self.label_style.configure("Default.TLabel", foreground="black", background=self._label_bgcolor,
                                   font=(self._font_family, self._font_size), anchor="center")

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
        for row in range(3):
            Grid.rowconfigure(self._root, row, weight=1)

        for col in range(7):
            Grid.columnconfigure(self._root, col, weight=1)

        # set size
        # Grid.rowconfigure(self._root, 3, minsize=80)
        Grid.rowconfigure(self._root, 0, minsize=self._model_plot_height)
        Grid.columnconfigure(self._root, 0, minsize=self._run_button_width)
        Grid.columnconfigure(self._root, 1, minsize=self._parameter_listbox_width)
        Grid.columnconfigure(self._root, 2, minsize=self._parameter_button_width)

        # set up window components

        # main panel
        self.button_run = ttk.Button(self._root,
                                      text=self._default_run_text,
                                      command=self.run_meta_search,
                                      style="Viz.TButton")

        self.button_stop = ttk.Button(self._root,
                                     text=self._default_stop_text,
                                     command=self.stop_meta_search,
                                     style="Viz.TButton")


        # self.model_plot_canvas = Canvas(self._root, width=280, height=250)
        self.model_plot_canvas = Label(self._root)
        self._model_plot_width = self._run_button_width + self._parameter_listbox_width + self._parameter_button_width

        # parameter control

        self.listbox_parameters = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                          bg = self._listbox_bgcolor)
        self.listbox_parameters.bind('<Double-Button>', self.modify_parameter)

        self.button_parameters_selection_up = ttk.Button(self._root,
                                             text="  /\\  ",
                                             command=self.parameters_selection_up,
                                             style="UpDown.TButton")

        self.button_parameters_selection_down = ttk.Button(self._root,
                                               text="  \\/  ",
                                               command=self.parameters_selection_down,
                                               style="UpDown.TButton")

        # performance plot

        self.listbox_performance = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                          bg=self._listbox_bgcolor)
        self.listbox_performance.bind('<<ListboxSelect>>', self.update_plot(plot_type=Plot_Windows.PERFORMANCE))

        self.button_performance_selection_up = ttk.Button(self._root,
                                                         text="  /\\  ",
                                                         command=self.performance_selection_up,
                                                         style="UpDown.TButton")

        self.button_performance_selection_down = ttk.Button(self._root,
                                                           text="  \\/  ",
                                                           command=self.performance_selection_down,
                                                           style="UpDown.TButton")

        self._fig_performance = Figure(figsize=(1, 1), dpi=100)
        self._axis_performance_line = self._fig_performance.add_subplot(111)
        self._axis_performance_line_3d = self._fig_performance.add_subplot(111)
        self._fig_performance.subplots_adjust(bottom=0.2)
        self._fig_performance.subplots_adjust(left=0.35)
        self._axis_performance_line.plot([0],[0])
        # self._axis_performance.scatter([0], [0], s=self._scatter_area, facecolors='none', edgecolors=self._scatter_color)
        self._axis_performance_line.set_xlabel('Ordinate', fontsize=self._font_size)
        self._axis_performance_line.set_ylabel('Epochs', fontsize=self._font_size)
        self._axis_performance_line.set_title('No Data Available', fontsize=self._font_size)
        self._axis_performance_line.grid()
        self._canvas_performance = FigureCanvasTkAgg(self._fig_performance, self._root)


        # supplementary plot

        self.listbox_supplementary = Listbox(self._root, selectmode=SINGLE, font=(self._font_family, self._font_size),
                                           bg=self._listbox_bgcolor)
        self.listbox_supplementary.bind('<<ListboxSelect>>', self.update_plot(plot_type=Plot_Windows.SUPPLEMENTARY))

        self.button_supplementary_selection_up = ttk.Button(self._root,
                                                          text="  /\\  ",
                                                          command=self.supplementary_selection_up,
                                                          style="UpDown.TButton")

        self.button_supplementary_selection_down = ttk.Button(self._root,
                                                            text="  \\/  ",
                                                            command=self.supplementary_selection_down,
                                                            style="UpDown.TButton")

        self._fig_supplementary= Figure(figsize=(1, 1), dpi=100)
        self._axis_supplementary_line = self._fig_supplementary.add_subplot(111)
        self._fig_supplementary.subplots_adjust(bottom=0.2)
        self._fig_supplementary.subplots_adjust(left=0.35)
        self._axis_supplementary_line.plot([0], [0])
        # self._axis_performance.scatter([0], [0], s=self._scatter_area, facecolors='none', edgecolors=self._scatter_color)
        self._axis_supplementary_line.set_xlabel('Epochs', fontsize=self._font_size)
        self._axis_supplementary_line.set_ylabel('Ordinate', fontsize=self._font_size)
        self._axis_supplementary_line.set_title('No Data Available', fontsize=self._font_size)
        self._axis_supplementary_line.grid()
        self._canvas_supplementary = FigureCanvasTkAgg(self._fig_supplementary, self._root)

        self.init_window()

    def init_window(self):

        # set up GUI
        self._root.title(self._title)

        # main panel
        self.button_run.grid(row=2, column=0, sticky=N + S + E + W)
        self.button_stop.grid(row=1, column=0, sticky=N + S + E + W)
        self.model_plot_canvas.grid(row=0, column=0, columnspan=3, sticky=N + S + E + W)

        # parameter control
        self.listbox_parameters.grid(row=1, rowspan=2, column=1, sticky=N + S + E + W)
        self.button_parameters_selection_up.grid(row=1, column=2, sticky=N + S + E + W)
        self.button_parameters_selection_down.grid(row=2, column=2, sticky=N + S + E + W)

        # performance plot
        self.listbox_performance.grid(row=1, rowspan=2, column=3, sticky=N + S + E + W)
        self.button_performance_selection_up.grid(row=1, column=4, sticky=N + S + E + W)
        self.button_performance_selection_down.grid(row=2, column=4, sticky=N + S + E + W)
        self._canvas_performance.get_tk_widget().grid(row=0, column=3, columnspan=2, sticky=N + S + E + W)

        # supplementary plot
        self.listbox_supplementary.grid(row=1, rowspan=2, column=5, sticky=N + S + E + W)
        self.button_supplementary_selection_up.grid(row=1, column=6, sticky=N + S + E + W)
        self.button_supplementary_selection_down.grid(row=2, column=6, sticky=N + S + E + W)
        self._canvas_supplementary.get_tk_widget().grid(row=0, column=5, columnspan=2, sticky=N + S + E + W)

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

    def parameters_selection_up(self):

        self.move_listbox_selection(self.listbox_parameters, -1)
        # self.set_DV_name()

    def parameters_selection_down(self):

        self.move_listbox_selection(self.listbox_parameters, +1)
        # self.set_DV_name()

    def performance_selection_up(self):

        self.move_listbox_selection(self.listbox_performance, -1)
        # self.set_DV_name()

    def performance_selection_down(self):

        self.move_listbox_selection(self.listbox_performance, +1)
        # self.set_DV_name()

    def supplementary_selection_up(self):

        self.move_listbox_selection(self.listbox_supplementary, -1)
        # self.set_DV_name()

    def supplementary_selection_down(self):

        self.move_listbox_selection(self.listbox_supplementary, +1)
        self.set_DV_name()


    def set_DV_name(self, name=None):
        pass

        # if name is None:
        #     if len(self.listbox_DVs.curselection()) > 0:
        #         index = int(self.listbox_DVs.curselection()[0])
        #         value = self.listbox_DVs.get(index)
        #         for DV in self._exp.DVs:
        #             if DV.get_variable_label() == value:
        #                 self._DV_name = DV.get_name()
        # else:
        #     self._DV_name = name
        #
        # self.update_plot_button()

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
                                            + str(num_meta_idx), command=self.resume_meta_search)
            else:
                self.button_run.config(text="PAUSE\n"
                                            + str(meta_idx)
                                            + " (" + str_percent + "%)"
                                            + " / "
                                            + str(num_meta_idx), command=self.pause_meta_search)
        else:
            self.button_run.config(text=self._default_run_text, command=self.run_meta_search)

    def update_model_plot(self):
        # load image
        model_image_path = self.theorist.plot_model(self.object_of_study)
        image = Image.open(model_image_path)

        # resize image
        width, height = image.size
        resized_width = width * (self._model_plot_height/height)
        resized_height = height * (self._model_plot_width/width)
        if resized_width > self._model_plot_width:
            width = self._model_plot_width
            height = resized_height
        elif resized_height > self._model_plot_height:
            width = resized_width
            height = self._model_plot_height
        else:
            width = self._model_plot_width
            height = resized_height
        image = image.resize((int(width), int(height)), Image.ANTIALIAS)

        # draw image
        img = ImageTk.PhotoImage(image)
        self.model_plot_canvas.config(image = img)

        # needed here, otherwise canvas doesn't update
        self._root.update()

    def update_parameter_list(self, model_search_parameters):
        self.listbox_parameters.delete(0, 'end')
        keys = model_search_parameters.keys()
        for key in keys:
            param_label = key + " = " + str(model_search_parameters[key][0])
            self.listbox_parameters.insert(END, param_label)

    def update_performance_plot_list(self, performance_plots):
        self.listbox_performance.delete(0, 'end')
        keys = performance_plots.keys()
        for key in keys:
            param_label = key
            self.listbox_performance.insert(END, param_label)

    def update_supplementary_plot_list(self, supplementary_plots):
        self.listbox_supplementary.delete(0, 'end')
        keys = supplementary_plots.keys()
        for key in keys:
            param_label = key
            self.listbox_supplementary.insert(END, param_label)

    def modify_parameter(self, click):
        listbox_selection = self.listbox_parameters.curselection()[0]
        listbox_value = str(self.listbox_parameters.get(listbox_selection))
        key = listbox_value.split(" =")[0]

        model_search_parameters = self.theorist.get_model_search_parameters()
        if key in model_search_parameters.keys():
            if model_search_parameters[key][1] is True: # parameter is modifiable
                new_param_value = simpledialog.askstring(key, "Please enter new value:",
                                                         parent=self._root)
                if new_param_value is not None:
                    self.theorist.set_model_search_parameter(key, new_param_value)
            else:
                messagebox.showinfo(key, "Parameter cannot be altered while running.")

        model_search_parameters = self.theorist.get_model_search_parameters()
        self.update_parameter_list(model_search_parameters)

    def update_plot(self, plot_type=None, plots=None, save=False, plot_name=""):

        if plot_type == Plot_Windows.PERFORMANCE:
            relevant_listbox = self.listbox_performance

            if isinstance(plots, dict) is False:
                plots = self.theorist.get_performance_plots(self.object_of_study)

            if hasattr(self, '_axis_performance_line'):
                plot_axis = self._axis_performance_line

            if hasattr(self, '_canvas_performance'):
                plot_canvas = self._canvas_performance

        elif plot_type == Plot_Windows.SUPPLEMENTARY:
            relevant_listbox = self.listbox_supplementary

            if isinstance(plots, dict) is False:
                plots = self.theorist.get_supplementary_plots(self.object_of_study)

            if hasattr(self, '_axis_supplementary_line'):
                plot_axis = self._axis_supplementary_line

            if hasattr(self, '_canvas_supplementary'):
                plot_canvas = self._canvas_supplementary

        else:
            return

        listbox_selection = relevant_listbox.curselection()
        if len(listbox_selection) == 0:
            listbox_selection = [0]
        key = str(relevant_listbox.get(listbox_selection[0]))

        # during initial call, key is empty
        if key == '':
            return

        if key in plots.keys():
            plot_dict = plots[key]

            type = plot_dict[config.plot_key_type]

            if plot_type == Plot_Windows.PERFORMANCE:

                if type == Plot_Types.SURFACE_SCATTER:
                    self._switch_performance_3d_plot()
                else:
                    self._switch_performance_2d_plot()

                if hasattr(self, '_axis_performance_line'):
                    plot_axis = self._axis_performance_line

                if hasattr(self, '_canvas_performance'):
                    plot_canvas = self._canvas_performance

            elif plot_type == Plot_Windows.SUPPLEMENTARY:

                if hasattr(self, '_axis_supplementary_line'):
                    plot_axis = self._axis_supplementary_line

                if hasattr(self, '_canvas_supplementary'):
                    plot_canvas = self._canvas_supplementary


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
                for idx, (x, y, leg) in enumerate(zip(x_data, y_data, legend)):
                    plots.append(plot_axis.plot(x, y, self._plot_colors[idx], label=leg))

                # adjust axes
                plot_axis.set_xlim(x_limit[0], x_limit[1])
                plot_axis.set_ylim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x_label, fontsize=self._plot_fontSize)
                plot_axis.set_ylabel(y_label, fontsize=self._plot_fontSize)

                plot_axis.legend(loc=2, fontsize="small")

                plot_canvas.draw()

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

                # generate plots
                plot_axis.cla()
                del plot_axis.lines[:]  # remove previous lines
                plots = list()
                # plot data
                plots.append(plot_axis.scatter(x_data, y_data, marker='.', c='r', label=legend[0]))

                # plot model prediction
                plots.append(plot_axis.plot(x_model, y_model, 'k', label=legend[1]))

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

                # generate plots
                plot_axis.cla()
                plots = list()

                # plot data
                plots.append(plot_axis.scatter(x1_data, x2_data, y_data, color = (1, 0, 0, 0.5), label=legend[0]))
                # plot model prediction
                plots.append(plot_axis.plot_trisurf(x1_model, x2_model, y_model, color=(0, 0, 0, 0.5), label=legend[1]))

                # adjust axes
                plot_axis.set_xlim(x1_limit[0], x1_limit[1])
                plot_axis.set_ylim(x2_limit[0], x2_limit[1])
                plot_axis.set_zlim(y_limit[0], y_limit[1])

                # set labels
                plot_axis.set_xlabel(x1_label, fontsize=self._plot_fontSize)
                plot_axis.set_ylabel(x2_label, fontsize=self._plot_fontSize)
                plot_axis.set_zlabel(y_label, fontsize=self._plot_fontSize)

            # finalize performance plot
            plot_axis.set_title(key, fontsize=self._plot_fontSize)
            plot_canvas.draw()
            if save is True:
                if plot_name is not "":
                    plot_filepath = os.path.join(self.theorist.results_plots_path, 'plot_' + plot_name + '_' + key + '.png')
                else:
                    plot_filepath = os.path.join(self.theorist.results_plots_path, 'plot_' + key + '.png')
                if plot_type == Plot_Windows.PERFORMANCE:
                    self._fig_performance.savefig(plot_filepath)
                else:
                    self._fig_supplementary.savefig(plot_filepath)



        else:
            raise Exception("Key '" + str(key) + "' not found in dictionary performance_plots.")

    def _switch_performance_2d_plot(self):
        if self._2d_plot:
            return
        self._fig_performance = Figure(figsize=(1, 1), dpi=100)
        self._axis_performance_line = self._fig_performance.add_subplot(111)
        self._axis_performance_line.plot([0], [0])
        self._axis_performance_line.set_xlabel('Ordinate', fontsize=self._font_size)
        self._axis_performance_line.set_ylabel('Epochs', fontsize=self._font_size)
        self._axis_performance_line.set_title('No Data Available', fontsize=self._font_size)
        self._configure_performance_plot()
        self._2d_plot = True

    def _switch_performance_3d_plot(self):
        if not self._2d_plot:
            return
        self._fig_performance = Figure(figsize=(1, 1), dpi=100)
        self._axis_performance_line = self._fig_performance.add_subplot(111, projection='3d')
        self._axis_performance_line.plot([0], [0], [0])
        self._axis_performance_line.set_xlabel('X', fontsize=self._font_size)
        self._axis_performance_line.set_ylabel('Y', fontsize=self._font_size)
        self._axis_performance_line.set_zlabel('Z', fontsize=self._font_size)
        self._axis_performance_line.set_title('No Data Available', fontsize=self._font_size)
        self._configure_performance_plot()
        self._2d_plot = False

    def _configure_performance_plot(self):
        self._fig_performance.subplots_adjust(bottom=0.2)
        self._fig_performance.subplots_adjust(left=0.35)
        self._axis_performance_line.grid()
        self._canvas_performance = FigureCanvasTkAgg(self._fig_performance, self._root)
        self._canvas_performance.get_tk_widget().grid(row=0, column=3, columnspan=2, sticky=N + S + E + W)
        self._root.update()

    def stop_meta_search(self):
        self._running = False
        if self._paused is True:
            self.reset_gui()

    def pause_meta_search(self):
        self._paused = True

    def resume_meta_search(self):
        self._paused = False
        self.run_meta_search(resume=True)

    def run_meta_search(self, resume=False):

        if resume is False:
            self._running = True
            self._paused = False
            self.update_run_button()
            self._root.update()

            # initialize meta-parameter search
            self.theorist.init_meta_search(self.object_of_study)

        # perform architecture search for different hyper-parameters
        for idx, meta_params in enumerate(self.theorist._meta_parameters):

            if resume is True:
                if idx < self._last_meta_param_idx:
                    continue
            else:

                [arch_weight_decay_df, num_graph_nodes, seed] = meta_params
                self.theorist.init_model_search(self.object_of_study)

                meta_param_str = self.theorist._meta_parameters_to_str()

                # update model parameters
                model_search_parameters = self.theorist.get_model_search_parameters()
                self.update_parameter_list(model_search_parameters)

                # update performance plot list
                performance_plots = self.theorist.get_performance_plots(self.object_of_study)
                self.update_performance_plot_list(performance_plots)

                # update supplementary plot list
                supplementary_plots = self.theorist.get_supplementary_plots(self.object_of_study)
                self.update_supplementary_plot_list(supplementary_plots)

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

                # update model plot
                self.update_model_plot()

                # update parameter list
                model_search_parameters = self.theorist.get_model_search_parameters()

                # update performance plot
                self.theorist.log_plot_data(epoch, self.object_of_study)
                performance_plots  = self.theorist.get_performance_plots(self.object_of_study)
                self.update_plot(Plot_Windows.PERFORMANCE, performance_plots)

                # update supplementary plot
                supplementary_plots = self.theorist.get_supplementary_plots(self.object_of_study)
                self.update_plot(Plot_Windows.SUPPLEMENTARY, supplementary_plots)

                self.update_parameter_list(model_search_parameters)


            if self._running is True:
                self.theorist.log_model_search(self.object_of_study)

                # save all performance plots
                performance_plots = self.theorist.get_performance_plots(self.object_of_study)
                for item in range(self.listbox_performance.size()):
                    self.set_listbox_selection(self.listbox_performance, item)
                    plot_str = meta_param_str + "_search"
                    self.update_plot(Plot_Windows.PERFORMANCE, performance_plots, save=True, plot_name=plot_str)
                # save all supplementary plots
                supplementary_plots = self.theorist.get_supplementary_plots(self.object_of_study)
                for item in range(self.listbox_supplementary.size()):
                    self.set_listbox_selection(self.listbox_supplementary, item)
                    self.update_plot(Plot_Windows.SUPPLEMENTARY, supplementary_plots, save=True, plot_name=meta_param_str)

                # Theorist: evaluate model architecture

                # initialize meta evaluation
                self.theorist.init_meta_evaluation(self.object_of_study)

                # perform architecture search for different hyper-parameters
                for eval_meta_params in self.theorist._eval_meta_parameters:

                    eval_param_str = self.theorist._eval_meta_parameters_to_str()

                    self.update_run_button(meta_idx=idx+1, num_meta_idx=len(self.theorist._eval_meta_parameters))

                    self.theorist.init_model_evaluation(self.object_of_study)
                    # loop over epochs
                    for epoch in range(self.theorist.eval_epochs):
                        # run single epoch
                        self.theorist.run_eval_epoch(epoch, self.object_of_study)
                        # log performance (for plotting purposes)
                        self.theorist.log_plot_data(epoch, self.object_of_study)
                        performance_plots = self.theorist.get_performance_plots(self.object_of_study)
                        self.update_plot(Plot_Windows.PERFORMANCE, performance_plots)
                        self._root.update()
                        self.update_run_button(epoch=epoch + 1, num_epochs=self.theorist.eval_epochs,
                                               meta_idx=idx + 1, num_meta_idx=len(self.theorist._eval_meta_parameters))

                    # save all performance plots
                    performance_plots = self.theorist.get_performance_plots(self.object_of_study)
                    for item in range(self.listbox_performance.size()):
                        self.set_listbox_selection(self.listbox_performance, item)
                        plot_str = meta_param_str + "_eval_" + eval_param_str
                        self.update_plot(Plot_Windows.PERFORMANCE, performance_plots, save=True, plot_name=plot_str)

                    # log model evaluation
                    self.theorist.log_model_evaluation(self.object_of_study)

                # sum up meta evaluation
                self.theorist.log_meta_evaluation(self.object_of_study)

                self.theorist._meta_parameters_iteration += 1

        if self._running is True:
            best_model = self.theorist.get_best_model(self.object_of_study, plot_model=True)

            # # save all performance plots
            # self.theorist.model = best_model
            # performance_plots = self.theorist.get_performance_plots(self.object_of_study)
            # for item in range(self.listbox_performance.size()):
            #     self.set_listbox_selection(self.listbox_performance, item)
            #     plot_str = meta_param_str + "_eval_"
            #     self.update_plot(Plot_Windows.PERFORMANCE, performance_plots, save=True, plot_name=plot_str)

        else:
            # reset gui elements
            self.reset_gui()

        self._running = False


    def reset_gui(self):
        self.update_run_button()
        self.listbox_parameters.delete(0, 'end')
        self.listbox_performance.delete(0, 'end')
        self.listbox_supplementary.delete(0, 'end')
        self._root.update()