import socket
import os.path
import experiment_config as config
import file_upload_server_protocol as protocol
import time

class File_Upload_Client():

    HOST = config.HOST_IP       # The server's hostname or IP address
    PORT = config.HOST_PORT     # The port used by the server

    GUI = None

    def __init__(self, HOST=None, PORT=None, GUI=None):

        if HOST is not None:
            self.HOST = HOST

        if PORT is not None:
            self.PORT = PORT

        if GUI is not None:
            self.GUI = GUI


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