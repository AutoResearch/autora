from tkinter import *
from tkinter import ttk
from AER_experimentalist.experiment_environment.utils import *
from AER_experimentalist.experiment_environment.experimentalist_GUI import Experimentalist_GUI
import AER_experimentalist.experiment_environment.experiment_config as config
from AER_experimentalist.experiment_environment.experiment_server import Experiment_Server
import threading, queue
import os

def runloop(gui=None, run_local=False):
    '''
    After result is produced put it in queue
    '''
    # result = 0
    # for i in range(10000000):
    #     # Do something with result
    #     # thread_queue.put(result)
    #     print(i)
    #     gui.set_label(i)
    if run_local is True:
        experiment_server = Experiment_Server(gui=gui)
    else:
        # host_name = socket.gethostname()
        # host_ip = socket.gethostbyname(host_name)
        host_ip = os.popen("hostname -I").readline()[:-2]
        experiment_server = Experiment_Server(gui=gui, host=host_ip)
    gui.server_running = True
    gui.experiment_server = experiment_server
    experiment_server.launch()


class Experiment_Server_GUI(Experimentalist_GUI):

    _start_server_bgcolor = "#fbfc9f"
    _stop_server_bgcolor = config.stop_bgcolor
    _default_label_status_text = "JOB STATUS"
    _default_label_experiment_status_text = "EXPERIMENT STATUS"

    _default_label_start_server = "START SERVER"
    _default_label_stop_server = "STOP SERVER"

    _experiments_path = config.server_path + config.experiments_path
    _sequences_path = config.server_path + config.sequences_path
    _data_path = config.server_path + config.data_path

    experiment_server = None
    server_running = False
    run_local = False

    # Initialize GUI.
    def __init__(self, root=None, path=None, run_local=False):

        super().__init__(root=root, path=path)

        self.run_local = run_local

        self.start_server_button_style = ttk.Style()
        self.start_server_button_style.configure("ServerStart.TButton", foreground="black",
                                            background=self._start_server_bgcolor,
                                            font=(self._font_family, self._font_size_button))

        self.stop_server_button_style = ttk.Style()
        self.stop_server_button_style.configure("ServerStop.TButton", foreground="black",
                                                 background=self._stop_server_bgcolor,
                                                 font=(self._font_family, self._font_size_button))


        # set up window components

        self.button_server = ttk.Button(self._root,
                                     text=self._default_label_start_server,
                                     command=self.toggle_server,
                                     style="ServerStart.TButton")

        self.button_run = ttk.Button(self._root,
                                     text=self._default_label_experiment_status_text,
                                     style="Run.TButton")

        self.button_server.grid(row=3, column=0, columnspan=2, sticky=N+S+E+W)
        self.button_run.grid(row=0, column=6, sticky=N + S + E + W)

        # experiment file label
        self.label_status.config(text=self._default_label_status_text)

        # grid adjustments
        Grid.columnconfigure(self._root, 6, minsize=100)
        Grid.columnconfigure(self._root, 0, minsize=200)

    def set_label(self, value):
        self.label_status.config(text=str(value))

    def update_status(self, status, msg):
        self.label_status.config(text=msg)
        self.listbox_status.insert(END, msg)
        self.set_listbox_selection(self.listbox_status, self.listbox_status.size() - 1)
        self.listbox_status.yview_moveto(1)

    def listen_for_result(self):
        '''
        Check if there is something in the queue
        '''
        try:
            self.res = self.thread_queue.get(0)
            print('loop terminated')
        except queue.Empty:
            self.after(100, self.listen_for_result)

    def toggle_server(self):
        '''
        Spawn a new thread for running long loops in background
        '''
        if self.experiment_server is None and self.server_running is False:
            # clear status listbox
            self.listbox_status.delete(0, 'end')
            # launch server
            self.thread_queue = queue.Queue()
            self.new_thread = threading.Thread(
                target=runloop,
                kwargs={'gui':self, 'run_local':self.run_local})
            self.new_thread.start()
            self.after(100, self.listen_for_result)
            # change server
            self.button_server.config(text=self._default_label_stop_server, style="ServerStop.TButton")
        else:
            self.experiment_server.abort()
            self.experiment_server = None
            self.server_running = False
            self.button_server.config(text=self._default_label_start_server, style="ServerStart.TButton")

    def init_run(self):
        self.button_run.configure(text=self._default_label_experiment_status_text, style="Run.TButton")

    def init_STOP_button(self):
        self.button_run.configure(text="RUNNING", style="Stop.TButton")

    def update_STOP_button(self, progress):
        self.button_run.configure(text="RUNNING" + "(" + str(round(progress)) + "%)")

    def close_server(self):
        if self.experiment_server is not None:
            self.experiment_server.abort()
            self.experiment_server = None
            self.server_running = False
            self.button_server.config(text=self._default_label_start_server, style="ServerStart.TButton")
