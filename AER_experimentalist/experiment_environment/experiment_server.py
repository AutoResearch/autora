
import socket
import os.path
import AER_experimentalist.experiment_environment.experiment_config as config
import AER_experimentalist.experiment_environment.client_server_protocol as protocol
from AER_experimentalist.experiment_environment.client_server_interface import Client_Server_Interface
# from experiment_server_GUI import Experiment_Server_GUI

class Experiment_Server(Client_Server_Interface):

    def __init__(self, session_ID=None, host=None, port=None, gui=None, exp=None):

        super().__init__(session_ID, host, port, gui)

        self._abort = False
        self._data_collected = False

        self.exp_folder_path = config.experiments_path
        self.seq_folder_path = config.sequences_path
        self.data_folder_path = config.data_path
        self.session_folder_path = config.session_path

        self.main_directory = config.server_path

        self.exp = exp

    def launch(self):

        # create socket instance
        # AF_INET is the Internet address family for IPv4
        # SOCK_STREAM is the socket type for TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._s:

            # associate the socket with a specific network interface and port number
            self._s.bind((self.host, self.port))

            #  enables server to accept() connections
            self._s.listen()

            # keep listening for connections until server is aborted
            while True:
                self._print_status(protocol.STATUS_LISTENING, "Listening for client.")
                self._print_status(protocol.STATUS_LISTENING, self.host + ":" + str(self.port))

                # get connection
                self._socket, self._addr = self._s.accept()

                with self._socket:

                    # update GUI
                    self._print_status(protocol.STATUS_CONNECTED, "Connected by " + str(self._addr) + ".")

                    while True:
                        data = self._socket.recv(1024)

                        if data != b'':
                            print(repr(data))

                        # if a connection is established, get the session ID
                        # the protocol cannot continue without identifying the session ID between client and server

                        if data == protocol.CLEAR_SESSIONS:
                            self._set_status(protocol.STATUS_CLEARING_SESSIONS)
                            self._send_confirmation()
                            self._clear_sessions(config.server_path)
                            self._print_status(protocol.STATUS_CLEARING_SESSIONS, "All sessions cleared.")

                        elif  data == protocol.TRANSFER_SESSION_ID:
                            self._set_status(protocol.STATUS_TRANSFERRING_SESSION_ID)
                            self._send_confirmation()
                            self._receive_session_ID()
                            self._print_status(protocol.STATUS_RECEIVED_SESSION_ID,
                                              "Received session ID: " + str(self.session_ID) + ".")

                        elif data == protocol.NEGOTIATE_STATUS:
                            self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Negotiating status.")
                            self._socket.sendall(protocol.OK)
                            self._negotiate_status()
                            if self.job_status == protocol.JOB_STATUS_COMPLETED_JOB:
                                break

                        elif data == protocol.INITIATE_TRANSFER:

                            self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Initiated file transfer.")
                            self._socket.sendall(protocol.OK)
                            self._receive_file()

                        elif data == protocol.START_EXPERIMENT:
                            self._print_status(protocol.STATUS_REQUESTING_EXP, "Received experiment request.")
                            self._socket.sendall(protocol.OK)
                            # collecting experiment name
                            while True:
                                data = self._socket.recv(1024)

                                if data == protocol.TRANSFER_FILEPATH:
                                    self._print_status(protocol.STATUS_RECEIVING_FILE_PATH, "Receiving file path.")
                                    self._socket.sendall(protocol.OK)

                                    file_path = self._socket.recv(1024).decode(protocol.STRING_FORMAT)
                                    self._print_status(protocol.STATUS_RECEIVING_FILE_PATH,
                                                       file_path)
                                    self._socket.sendall(protocol.OK)
                                    self.exp_file_path = file_path
                                    break

                            self._run_experiment()

                        elif data == protocol.REQUEST_COMPLETE:
                            self._print_status(protocol.STATUS_COMPLETED_TRANSFER, "File transfer complete.")
                            self._socket.sendall(protocol.OK)
                            self._terminate_session()
                            break

                        if self.job_status == protocol.JOB_STATUS_COMPLETED_EXPERIMENT:
                            if self._data_collected is True:
                                self._send_data_file()

                        if self._abort is True:
                            break

                if self._abort is True:
                    break

            self._close_server()

    def abort(self):
        self._abort = True
        self._print_status(protocol.STATUS_ABORT, "Aborted.")
        # need to connect to server once to make it close
        try:
            socket.socket(socket.AF_INET,
                          socket.SOCK_STREAM).connect((self.host, self.port))
        except Exception as e:
            self._print_status(protocol.STATUS_ERROR, "Error:" + str(e))


    def _close_server(self):
        self._s.close()
        self._print_status(protocol.STATUS_ABORT, "Server Closed.")

    def _set_session_ID(self, session_ID):
        self.session_ID = session_ID
        self.session_file_path = self.session_folder_path + str(self.session_ID) + '.session'

    def _receive_session_ID(self):

        while True:

            data = self._socket.recv(1024)

            if data != b'':
                session_ID = data.decode(protocol.STRING_FORMAT)
                self._set_session_ID(session_ID)
                self._socket.sendall(protocol.OK)
                break

        self._set_status(protocol.STATUS_CONNECTED)
        self._load_job_status()


    def _print_status(self, status, msg):
        if self.gui is not None:
            self.gui.update_status(status, msg)
        else:
            print("Server: " + msg)

    def _negotiate_status(self):

        # initialization
        while True:
            data = self._socket.recv(1024)

            if data == protocol.TRANSFER_CLIENT_STATUS:
                self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Receiving client status...")
                self._socket.sendall(protocol.OK)
                break

        # receiving client status
        client_status = None

        while True:
            data = self._socket.recv(1024)

            if data != b'':
                data = data.decode(protocol.STRING_FORMAT)
                client_status = int(data)
                self._socket.sendall(protocol.OK)
                break

        self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Received job status: " + str(client_status))
        # negotiating status
        if self.job_status is not None:
            new_job_status = self.job_status # min(self.job_status, client_status)
            if self.data_file_path is not None:
                if os.path.exists(self.main_directory + self.data_file_path) is True:
                    self._data_collected = True
        else:
            new_job_status = client_status

        if new_job_status == protocol.JOB_STATUS_REQUESTED_EXPERIMENT: # need to go one step back
            new_job_status = protocol.JOB_STATUS_SENT_SEQUENCE_FILE
        self._set_job_status(new_job_status)

        # send new server status
        self._send_and_confirm(protocol.TRANSFER_SERVER_STATUS)
        self._send_and_confirm(bytes(str(self.job_status), 'utf-8'))

        # save status
        self._save_job_status()
        self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Negotiated job status: " + str(self.job_status))

    def _send_data_file(self):
        self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Sending data file...")
        self._send_file(self.data_file_path)
        print("serv: here 1")
        self._set_job_status(protocol.JOB_STATUS_COMPLETED_JOB)
        self._save_job_status()


    def _run_experiment(self):
        self._set_job_status(protocol.JOB_STATUS_REQUESTED_EXPERIMENT)
        self._save_job_status()

        exp_file_name = os.path.basename(self.exp_file_path)
        if self.gui is not None:
            # need to pass experiment file name to gui
            self._print_status(protocol.STATUS_INITIATED_TRANSFER, exp_file_name)
            self.gui.load_experiment(exp_file_name)
            self.data_file_path = self.gui.run_experiment(plot=True)
        else:
            if self.exp is None:
                raise Exception("Cannot run experiment, no experiment specified.")
            else:
                experiments_path = config.server_path + config.experiments_path
                exp_path = os.path.join(experiments_path, exp_file_name)
                self.exp.load_experiment(exp_path)
                self.exp.run_experiment()
                data_file_path_tmp = config.server_path + self.exp._data_path
                print("data_file_path_tmp: " + data_file_path_tmp)
                self.exp.data_to_csv(data_file_path_tmp)
                self.data_file_path = self.exp._data_path

        # wrap up experiment
        self._wrap_up_experiment(self.data_file_path)

            # TODO: leaving this for test purposes
            # self.data_file_path = self.data_folder_path + 'experiment1_data.csv'
            # self._wrap_up_experiment(self.data_file_path)

    def _wrap_up_experiment(self, data_file_path):
        self._print_status(protocol.STATUS_REQUESTING_EXP, "Wrapping up experiment...")
        self.data_file_path = data_file_path
        self._data_collected = True
        self._set_job_status(protocol.JOB_STATUS_COMPLETED_EXPERIMENT)
        self._save_job_status()

    def _terminate_session(self):
        self._print_status(protocol.STATUS_TERMINATING_SESSION, "Terminating session...")
        self.exp_file_path = ''
        self.seq_file_path = ''
        self.data_file_path = ''
        self.session_ID = None
        self._data_collected = False


# upload_server = Experiment_Server()
# upload_server.launch()