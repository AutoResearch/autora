from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_oled_128x64_v2 import BrickletOLED128x64V2

class OLED_Output():

    _UID = "NNa"
    _HOST = "localhost"
    _PORT = 4223

    _MAX_LINES = 7

    _messages = list()

    # Initializes OLED device.
    def __init__(self, UID = None, HOST = None, PORT = None):

        if UID is not None:
            self._UID = UID

        if HOST is not None:
            self._HOST = HOST

        if PORT is not None:
            self._PORT = PORT

        self._ipcon = IPConnection()  # Create IP connection
        self._oled = BrickletOLED128x64V2(self._UID, self._ipcon)

        self._ipcon.connect(self._HOST, self._PORT)  # Connect to brickd

        # Clear display
        self._oled.clear_display()

    # Show message on OLED display.
    def show(self, messages = None, line = 0, position = 0):

        # clear display
        self._oled.clear_display()

        init_line = line

        if messages is None:
            messages = self._messages

        # loop through all messages
        for msg in messages:
            self._oled.write_line(init_line, position, msg)
            init_line += 1

    # Clear list of messages.
    def clear_messages(self):
        self._messages.clear()

    # Append a message.
    def append_message(self, messages):

        if isinstance(messages, str):
            self._messages.append(messages)
        else:
            for msg in messages:
                self._messages.append(msg)

        while len(self._messages) > self._MAX_LINES:
            self._messages.pop(0)

    # Append a message and show on display.
    def append_and_show_message(self, msg):
        self.append_message(msg)
        self.show()

    # Clear display.
    def clear_display(self):
        self._oled.clear_display()

    # Clean up measurement device.
    def clean_up(self):
        self._oled.clear_display()
        self._ipcon.disconnect()
