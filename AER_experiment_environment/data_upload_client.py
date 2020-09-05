import socket
import experiment_config as config
import file_upload_server_protocol as protocol
from file_upload_client import File_Upload_Client

class Data_Upload_Client(File_Upload_Client):

    data_path = ""       # path to data file

    _data_folder = config.data_path

    def __init__(self, HOST=None, PORT=None, GUI=None, data_file_name=None):

        super().__init__(HOST, PORT, GUI)

        if data_file_name is not None:
            self.data_path = self._data_folder + data_file_name

    def send_job(self, data_file_name = None):

        if data_file_name is not None:
            self.data_path = self._data_folder + data_file_name

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._s:

            try:
                self._s.connect((self.HOST, self.PORT))
            except:
                self.print_status(protocol.CLIENT_STATUS_ERROR, "Could not connect to " + str(self.HOST) + ".")
                return False

            self.print_status(protocol.CLIENT_STATUS_CONNECTED, "Connected to " + str(self.HOST) + ".")

            # send experiment file
            self.print_status(protocol.CLIENT_STATUS_INITIATED_TRANSFER, "Sending data file...")
            self._send_file(self.data_path)

            # finalize interaction
            self._send_and_confirm(protocol.REQUEST_COMPLETE)
            self.print_status(protocol.CLIENT_STATUS_COMPLETED_JOB, "Completed job.")

        return True


exp_client = Data_Upload_Client()
exp_client.send_job("experiment2_data.csv")