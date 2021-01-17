import socket
import AER_experimentalist.experiment_environment.experiment_config as config
import AER_experimentalist.experiment_environment.client_server_protocol as protocol
from AER_experimentalist.experiment_environment.client_server_interface import Client_Server_Interface
from os import path

class Experiment_Client(Client_Server_Interface):

    def __init__(self, session_ID, host=None, port=None, gui=None, experiment_file_name=None):

        super().__init__(session_ID, host, port, gui)

        self.exp_folder_path = config.experiments_path
        self.seq_folder_path = config.sequences_path
        self.data_folder_path = config.data_path
        self.session_folder_path = config.session_path
        self.main_directory = config.client_path

        if experiment_file_name is not None:
            self.exp_file_path = self.exp_folder_path + experiment_file_name

        # update client path
        if path.exists(self.main_directory) is False:
            self.main_directory = "client data/"

    def submit_job(self, experiment_file_name = None, clear_sessions=False):

        if experiment_file_name is not None:
            self.exp_file_path = self.exp_folder_path + experiment_file_name

        # read sequence file from experiment file
        self._read_experiment_file()

        self._print_status(protocol.STATUS_CONNECTED, "Attempting to connect to " + str(self.host) + ":" + str(self.port))
        return self.connect_to_server(clear_sessions)

    def connect_to_server(self, clear_sessions=False):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._socket:

            try:
                self._socket.connect((self.host, self.port))
            except:
                self._print_status(protocol.STATUS_ERROR, "Could not connect to " + str(self.host) + ".")
                return False

            self._set_status(protocol.STATUS_CONNECTED)
            self._print_status(protocol.STATUS_CONNECTED, "Connected to " + str(self.host) + ".")

            self._request_clear_sessions()

            # send session ID
            self._send_session_ID()

            # load current status
            self._load_job_status()
            self._negotiate_status()

            try:
                while True:

                    if self.job_status == protocol.JOB_STATUS_CONNECTED:
                        self._send_experiment_file()

                    elif self.job_status == protocol.JOB_STATUS_SENT_EXPERIMENT_FILE:
                        self._send_sequence_file()

                    elif self.job_status == protocol.JOB_STATUS_SENT_SEQUENCE_FILE:
                        self._request_experiment_execution()

                    elif self.job_status == protocol.JOB_STATUS_REQUESTED_EXPERIMENT:
                        while True:
                            data = self._socket.recv(1024)
                            if data == protocol.INITIATE_TRANSFER:
                                self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Initiated file transfer.")
                                self._socket.sendall(protocol.OK)
                                self._receive_data_file()
                                break

                    elif self.job_status == protocol.JOB_STATUS_RECEIVED_DATA_FILE:
                        self._complete_session()
                        break

                    elif self.job_status == protocol.JOB_STATUS_COMPLETED_JOB:
                        break

            except Exception as e: # if something goes wrong, return false
                print("An exception occurred: " + str(e))
                return False

        return True

    def _request_experiment_execution(self):
        self._print_status(protocol.STATUS_REQUESTING_EXP, "Requesting experiment execution...")
        self._send_and_confirm(protocol.START_EXPERIMENT)

        # tell server experiment file
        self._print_status(protocol.STATUS_SENDING_FILE_PATH, "Sending file path...")
        self._send_and_confirm(protocol.TRANSFER_FILEPATH)
        destination_file_path = self.exp_file_path
        self._print_status(protocol.STATUS_SENDING_FILE_PATH, destination_file_path)
        self._send_and_confirm(bytes(destination_file_path, 'utf-8'))

        # update status
        self._set_job_status(protocol.JOB_STATUS_REQUESTED_EXPERIMENT)
        self._save_job_status()
        self._print_status(protocol.STATUS_REQUESTING_EXP, "Waiting for experiment data...")

    def _receive_data_file(self):
        self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Receiving data file...")
        self._receive_file()
        print("Received data file")
        self._set_job_status(protocol.JOB_STATUS_RECEIVED_DATA_FILE)
        self._save_job_status()

    def _send_experiment_file(self):
        self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Sending experiment file: " )
        self._print_status(protocol.STATUS_INITIATED_TRANSFER, self.exp_file_path)
        self._send_file(self.exp_file_path)
        self._set_job_status(protocol.JOB_STATUS_SENT_EXPERIMENT_FILE)
        self._save_job_status()

    def _send_sequence_file(self):
        self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Sending sequence file...")
        self._send_file(self.seq_file_path)
        self._set_job_status(protocol.JOB_STATUS_SENT_SEQUENCE_FILE)
        self._save_job_status()

    def _request_clear_sessions(self):
        self._print_status(protocol.STATUS_CLEARING_SESSIONS, "Clearing sessions...")
        self._send_and_confirm(protocol.CLEAR_SESSIONS)
        self._clear_sessions(self.main_directory)
        self._print_status(protocol.STATUS_CLEARING_SESSIONS, "All sessions cleared.")

    def _send_session_ID(self):
        self._print_status(protocol.STATUS_TRANSFERRING_SESSION_ID, "Sending session ID...")
        self._send_and_confirm(protocol.TRANSFER_SESSION_ID)
        self._send_and_confirm(bytes(str(self.session_ID), 'utf-8'))
        self._set_job_status(protocol.JOB_STATUS_CONNECTED)
        self._save_job_status()

    def _complete_session(self):
        self._print_status(protocol.JOB_STATUS_COMPLETED_JOB, "Finalizing session...")
        self._send_and_confirm(protocol.REQUEST_COMPLETE)
        self._set_status(protocol.JOB_STATUS_COMPLETED_JOB)
        self._save_job_status()

    def _set_session_ID(self, session_ID):
        self.session_ID = session_ID
        self.session_file_path = self.session_folder_path + str(self.session_ID) + '.session'

    def _print_status(self, status, msg):
        if self.gui is not None:
            self.gui.update_status(status, msg)
        else:
            print("Client: " + msg)

    def _negotiate_status(self):
        self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Negotiating job status...")
        self._send_and_confirm(protocol.NEGOTIATE_STATUS)

        # send client status
        self._send_and_confirm(protocol.TRANSFER_CLIENT_STATUS)
        self._send_and_confirm(bytes(str(self.job_status), 'utf-8'))
        self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Sent job status: " + str(self.job_status))

        # receive final status from server
        while True:
            data = self._socket.recv(1024)

            if data == protocol.TRANSFER_SERVER_STATUS:
                self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Receiving server status...")
                self._socket.sendall(protocol.OK)
                break

        while True:
            data = self._socket.recv(1024)

            if data != b'':
                data = data.decode(protocol.STRING_FORMAT)
                self._set_job_status(int(data))
                self._socket.sendall(protocol.OK)
                break

        # save status
        self._save_job_status()
        self._print_status(protocol.STATUS_NEGOTIATING_STATUS, "Negotiated job status:" + str(self.job_status))


    def _read_experiment_file(self, filepath=None):

        if filepath is not None:
            path = filepath
        else:
            path = self.exp_file_path

        # read experiment file
        file = open(self.main_directory + path, "r")
        for line in file:

            # read line of experiment file
            string = str(line)
            string = string.replace('\n', '')
            string = string.replace(' ', '')

            # read sequence file
            if (string.find('Sequence:') != -1):
                string = string.replace('Sequence:', '')
                self.seq_file_path = self.seq_folder_path + string


# session_ID = 1
# host = '192.168.188.27'
# port = 47778
# exp_client = Experiment_Client(session_ID, host=host, port=port)
# exp_client.submit_job("experiment2.exp", clear_sessions=True)