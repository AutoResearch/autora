import socket
import experiment_config as config
import file_upload_server_protocol as protocol
from file_upload_client import File_Upload_Client

class Experiment_Upload_Client(File_Upload_Client):

    exp_path = ""       # path to experiment file
    seq_path = ""       # path to sequence file

    _experiments_folder = config.experiments_path
    _sequences_folder = config.sequences_path

    def __init__(self, HOST=None, PORT=None, GUI=None, experiment_file_name=None):

        super().__init__(HOST, PORT, GUI)

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


exp_client = Experiment_Upload_Client()
exp_client.send_job("experiment2.exp")