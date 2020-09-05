#!/usr/bin/env python
# -*- coding: utf-8 -*-

HOST = "localhost"
PORT = 4223
UID = "M43" # Change XYZ to the UID of your RGB LED Button Bricklet

from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_rgb_led_button import BrickletRGBLEDButton
from tinkerforge.bricklet_oled_128x64_v2 import BrickletOLED128x64V2
from subprocess import call
from AER_epaper_offline import update_epaper
import time

# Callback function for button state changed callback
def shutdown_procedure():
        update_epaper()
        call("sudo shutdown -h now", shell=True)

ipcon = IPConnection() # Create IP connection

rlb = BrickletRGBLEDButton(UID, ipcon)  # Create device object

ipcon.connect(HOST, PORT)  # Connect to brickd
# Don't use device before ipcon is connected

# Set light blue color
rlb.set_color(255, 0, 0)

try:

    while True:
        # Get current button state
        state = rlb.get_button_state()

        if state == rlb.BUTTON_STATE_PRESSED:
            rlb.set_color(0, 0, 0)
            shutdown_procedure()

        time.sleep(5)

except:
    ipcon.disconnect()