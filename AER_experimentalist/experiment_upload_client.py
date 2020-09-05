import socket
import os.path
import experiment_config as config
import file_upload_server_protocol as protocol
import time
from file_upload_client import File_Upload_Client

class Experiment_Client(File_Upload_Client):

    HOST = config.HOST_IP       # The server's hostname or IP address
    PORT = config.HOST_PORT     # The port used by the server

    GUI = None

    exp_path = ""       # path to experiment file
    seq_path = ""       # path to sequence file

    _experiments_folder = config.experiments_path
    _sequences_folder = config.sequences_path

    def __init__(self, HOST=None, PORT=None, GUI=None, experiment_file_name=None):

        if HOST is not None:
            self.HOST = HOST

        if PORT is not None:
            self.PORT = PORT

        if GUI is not None:
            self.GUI = GUI

        if experiment_file_name is not None:
            self.exp_path = self._experiments_folder + experiment_file_name


    def send_job(self, experiment_file_name = None):

        if experiment_file_name is not None:
            self.exp_path = self._experiments_folder + experiment_file_name

        # read sequence file from experiment file
        self.read_experiment_file()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._s:

            try:
                self._s.connect((self.HOST, self.PORT))
            except:
                self.print_status(protocol.CLIENT_STATUS_ERROR, "Could not connect to " + str(self.HOST) + ".")
                return False

            self.print_status(protocol.CLIENT_STATUS_CONNECTED, "Connected to " + str(self.HOST) + ".")

            # send experiment file
            self.print_status(protocol.CLIENT_STATUS_INITIATED_TRANSFER, "Sending experiment file...")
            self._send_file(self.exp_path)

            # send sequence file
            self.print_status(protocol.CLIENT_STATUS_INITIATED_TRANSFER, "Sending sequence file...")
            self._send_file(self.seq_path)

            # finalize interaction
            self._send_and_confirm(protocol.REQUEST_COMPLETE)
            self.print_status(protocol.CLIENT_STATUS_COMPLETED_JOB, "Completed job.")

        return True

    def _send_file(self, filepath):

        # check if file exists
        if os.path.exists(filepath) is False:
            raise Exception("Cannot send file to server. File " + filepath + " does not exist.")

        # tell server to initiate file transfer
        self._send_and_confirm(protocol.INITIATE_TRANSFER)

        self.print_status(protocol.CLIENT_STATUS_INITIATED_TRANSFER, "Initiated file transfer.")

        # tell server relative file path
        self.print_status(protocol.CLIENT_STATUS_SENDING_FILE_PATH, "Sending file path...")
        self._send_and_confirm(protocol.TRANSFER_FILEPATH)
        if self.HOST == config.LOCAL_HOST:
            destination_filepath = filepath + "_received.exp"
        else:
            destination_filepath = filepath
        self.print_status(protocol.CLIENT_STATUS_SENDING_FILE_PATH, destination_filepath)
        self._send_and_confirm(bytes(destination_filepath, 'utf-8'))

        # send file
        self._send_and_confirm(protocol.TRANSFER_DATA)
        file = open(filepath, 'rb')
        line = file.read(1024)
        self.print_status(protocol.CLIENT_STATUS_SENDING_FILE_DATA, "Sending file...")
        while (line):
            self._send_and_confirm(line)
            line = file.read(1024)

        self._send_and_confirm(protocol.TRANSFER_COMPLETE)
        self.print_status(protocol.CLIENT_STATUS_COMPLETED_TRANSFER, "Completed file transfer.")

    def read_experiment_file(self, filepath=None):

        if filepath is not None:
            path = filepath
        else:
            path = self.exp_path

        # read experiment file
        file = open(path, "r")
        for line in file:

            # read line of experiment file
            string = str(line)
            string = string.replace('\n', '')
            string = string.replace(' ', '')

            # read sequence file
            if (string.find('Sequence:') != -1):
                string = string.replace('Sequence:', '')
                self.seq_path = self._sequences_folder + string


    def print_status(self, status, msg):
        if self.GUI is not None:
            self.GUI.set_status(status, msg)
        else:
            print("Client: " + msg)

    def _wait_for_server_OK(self):
        t0 = time.time()
        while True:
            reply = self._s.recv(1024)
            if reply == protocol.OK:
                break
            if (time.time() - t0) > protocol.TIMEOUT:
                self.print_status(self, protocol.CLIENT_STATUS_ERROR, "Timeout: no response from server.")
                break

    def _send_and_confirm(self, msg):
        self._s.sendall(msg)
        self._wait_for_server_OK()


exp_client = Experiment_Client()
exp_client.send_job("experiment2.exp")