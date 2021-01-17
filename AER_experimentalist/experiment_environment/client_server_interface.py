import socket
import os.path
import AER_experimentalist.experiment_environment.experiment_config as config
import AER_experimentalist.experiment_environment.client_server_protocol as protocol
import time

class Client_Server_Interface():


    def __init__(self, session_ID=None, host=None, port=None, gui=None):

        self.exp_file_path = ""  # path to experiment file
        self.seq_file_path = ""  # path to sequence file
        self.data_file_path = ""  # path to data file
        self.session_file_path = ""  # path to session file
        self.main_directory = ""  # path to main directory

        self.gui = None
        self.host = config.HOST_IP  # The server's hostname or IP address
        self.port = config.HOST_PORT  # The port used by the server

        self.job_status = None

        self.session_folder_path = config.session_path  # path to session folder

        if host is not None:
            self.host = host

        if port is not None:
            self.port = port

        if gui is not None:
            self.gui = gui

        if session_ID is not None:
            self._set_session_ID(session_ID)


    def _send_file(self, file_path):

        # check if file exists
        if os.path.exists(self.main_directory + file_path) is False:
            msg = "Cannot send file. File " + self.main_directory + file_path + " does not exist."
            self._print_status(protocol.STATUS_ERROR, msg)
            raise Exception(msg)

        # tell server to initiate file transfer
        self._send_and_confirm(protocol.INITIATE_TRANSFER)

        self._print_status(protocol.STATUS_INITIATED_TRANSFER, "Initiated file transfer.")

        # tell server relative file path
        self._print_status(protocol.STATUS_SENDING_FILE_PATH, "Sending file path...")
        self._send_and_confirm(protocol.TRANSFER_FILEPATH)
        if self.host == config.LOCAL_HOST:
            destination_file_path = file_path
        else:
            destination_file_path = file_path
        self._print_status(protocol.STATUS_SENDING_FILE_PATH, destination_file_path)
        self._send_and_confirm(bytes(destination_file_path, 'utf-8'))

        # send file
        self._send_and_confirm(protocol.TRANSFER_DATA)
        file = open(self.main_directory + file_path, 'rb')
        line = file.read(1024)
        self._print_status(protocol.STATUS_SENDING_FILE_DATA, "Sending file...")
        while (line):
            self._send_and_confirm(line)
            line = file.read(1024)

        self._send_and_confirm(protocol.TRANSFER_COMPLETE)
        print('after complete')
        self._print_status(protocol.STATUS_COMPLETED_TRANSFER, "Completed file transfer.")


    def _receive_file(self):

        file_path = ""

        while True:
            data = self._socket.recv(1024)

            if data == protocol.TRANSFER_FILEPATH:
                self._print_status(protocol.STATUS_RECEIVING_FILE_PATH, "Receiving file path.")
                self._socket.sendall(protocol.OK)

                file_path = self._socket.recv(1024)
                self._print_status(protocol.STATUS_RECEIVING_FILE_PATH, file_path.decode(protocol.STRING_FORMAT))
                self._socket.sendall(protocol.OK)

            elif data == protocol.TRANSFER_DATA:

                self._socket.sendall(protocol.OK)

                if file_path is "":
                    self._print_status(protocol.STATUS_ERROR,
                                      "File transfer failed: Did not receive file path.")
                    break

                try:
                    f = open(self.main_directory + file_path.decode(protocol.STRING_FORMAT), 'wb')
                    data = self._socket.recv(1024)
                    self._print_status(protocol.STATUS_RECEIVING_FILE_DATA, "Receiving file...")
                    while True: #data != protocol.TRANSFER_COMPLETE:
                        if data == protocol.TRANSFER_COMPLETE:
                            break
                        f.write(data)
                        print(repr(data))
                        self._socket.sendall(protocol.OK)
                        data = self._socket.recv(1024)
                    f.close()
                    self._socket.sendall(protocol.OK)
                    self._print_status(protocol.STATUS_RECEIVING_FILE_DATA, "Completed file transfer.")
                    break
                except:
                    self._print_status(protocol.STATUS_ERROR,
                                      "Could not write to file path: " + file_path)
                    break

    def _print_status(self, status, msg):
        if self.gui is not None:
            self.gui.set_status(status, msg)
        else:
            print(msg)

    def _wait_for_confirmation(self):
        t0 = time.time()
        while True:
            reply = self._socket.recv(1024)
            if reply == protocol.OK:
                break
            if (time.time() - t0) > protocol.TIMEOUT:
                self._print_status(protocol.STATUS_ERROR, "Timed out waiting for response")
                break

    def _send_confirmation(self):
        self._socket.sendall(protocol.OK)

    def _send_and_confirm(self, msg):
        self._socket.sendall(msg)
        self._wait_for_confirmation()

    def _set_status(self, status):
        self.status = status

    def _set_job_status(self, job_status):
        self.job_status = job_status

    def _set_session_ID(self, session_ID):
        self.session_ID = session_ID
        self.session_file_path = self.session_folder_path + str(self.session_ID) + '.session'

    def _save_job_status(self):

        f = open(self.main_directory + self.session_file_path, 'wt')

        if self.job_status is not None:
            f.write('Status:' + str(self.job_status) + '\n')
        if len(self.exp_file_path) > 0:
            f.write('ExperimentPath:' + str(self.exp_file_path) + '\n')
        if len(self.seq_file_path) > 0:
            f.write('SequencePath:' + str(self.seq_file_path) + '\n')
        if len(self.data_file_path) > 0:
            f.write('DataPath:' + str(self.data_file_path) + '\n')

        f.close()

    def _load_job_status(self):
        # read session file

        if os.path.isfile(self.main_directory + self.session_file_path):

            file = open(self.main_directory + self.session_file_path, "r")
            for line in file:

                # read line of experiment file
                string = str(line)
                string = string.replace('\n', '')

                # read file status
                if string.find('Status:') != -1:
                    string = string.replace('Status:', '')
                    self.job_status = int(string)

                # read experiment path
                if string.find('ExperimentPath:') != -1:
                    string = string.replace('ExperimentPath:', '')
                    self.exp_file_path = string

                # read sequence path
                if string.find('SequencePath:') != -1:
                    string = string.replace('SequencePath:', '')
                    self.seq_file_path = string

                # read data path
                if string.find('DataPath:') != -1:
                    string = string.replace('DataPath:', '')
                    self.data_file_path = string

        else:
            self.job_status = protocol.JOB_STATUS_CONNECTED

    def _clear_sessions(self, main_path=""):
        session_dir = main_path + self.session_folder_path
        file_list = [f for f in os.listdir(session_dir) if f.endswith(".session")]
        for f in file_list:
            os.remove(os.path.join(session_dir, f))