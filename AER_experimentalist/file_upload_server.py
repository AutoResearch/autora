
import socket
import os.path
import experiment_config as config
import experiment_server_protocol as protocol
from experiment_server_GUI import Experiment_Server_GUI

class Experiment_Server():

    HOST = config.HOST_IP  # The server's hostname or IP address
    PORT = config.HOST_PORT  # The port used by the server

    GUI = None

    _abort = False

    def __init__(self, HOST=None, PORT=None, GUI=None):

        if HOST is not None:
            self.HOST = HOST

        if PORT is not None:
            self.PORT = PORT

        if GUI is not None:
            self.GUI = GUI

    def launch(self):

        # create socket instance
        # AF_INET is the Internet address family for IPv4
        # SOCK_STREAM is the socket type for TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._s:

            # associate the socket with a specific network interface and port number
            self._s.bind((self.HOST, self.PORT))

            #  enables server to accept() connections
            self._s.listen()

            self.print_status(protocol.SERVER_STATUS_LISTENING, "Listening for client.")

            # get connection
            self._conn, self._addr = self._s.accept()
            with self._conn:
                print('Connected by', self._addr)

                # update GUI
                self.print_status(protocol.SERVER_STATUS_CONNECTED, "Connected by " + str(self._addr) + ".")

                while True:
                    data = self._conn.recv(1024)

                    if data != b'':
                        print(repr(data))

                    if data == protocol.INITIATE_TRANSFER:

                        self.print_status(protocol.SERVER_STATUS_INITIATED_TRANSFER, "Initiated file transfer.")
                        self._conn.sendall(protocol.OK)
                        self.receive_file()

                    elif data == protocol.REQUEST_COMPLETE:

                        self.print_status(protocol.SERVER_STATUS_COMPLETED_TRANSFER, "File transfer complete.")
                        self._conn.sendall(protocol.OK)

                    if self._abort is True:
                        self._close_server()
                        break


    def receive_file(self):

        file_path = ""

        while True:
            data = self._conn.recv(1024)

            if data == protocol.TRANSFER_FILEPATH:
                self.print_status(protocol.SERVER_STATUS_RECEIVING_FILE_PATH, "Receiving file path.")
                self._conn.sendall(protocol.OK)

                file_path = self._conn.recv(1024)
                self.print_status(protocol.SERVER_STATUS_RECEIVING_FILE_PATH, file_path.decode(protocol.STRING_FORMAT))
                self._conn.sendall(protocol.OK)

            if data == protocol.TRANSFER_DATA:

                self._conn.sendall(protocol.OK)

                if file_path is "":
                    self.print_status(protocol.SERVER_STATUS_ERROR,
                                      "File transfer failed: Did not receive file path.")
                    break

                try:
                    f = open(file_path, 'wb')
                    data = self._conn.recv(1024)
                    self.print_status(protocol.SERVER_STATUS_RECEIVING_FILE_DATA, "Receiving file...")
                    while data != protocol.TRANSFER_COMPLETE:
                        f.write(data)
                        print(repr(data))
                        self._conn.sendall(protocol.OK)
                        data = self._conn.recv(1024)
                    f.close()
                    self._conn.sendall(protocol.OK)
                    self.print_status(protocol.SERVER_STATUS_RECEIVING_FILE_DATA, "Completed file transfer.")
                    break
                except:
                    self.print_status(protocol.SERVER_STATUS_ERROR,
                                      "Could not write to file path: " + file_path)
                    break


    def abort(self):
        self._abort = True
        self.print_status(protocol.SERVER_STATUS_ABORT, "Aborted.")

    def _close_server(self):
        self._s.shutdown()
        self._s.close()
        self.print_status(protocol.SERVER_STATUS_ABORT, "Server Closed.")

    def print_status(self, status, msg):
        if self.GUI is not None:
            self.GUI.set_status(status, msg)
        else:
            print("Server: " + msg)


exp_server = Experiment_Server()
exp_server.launch()