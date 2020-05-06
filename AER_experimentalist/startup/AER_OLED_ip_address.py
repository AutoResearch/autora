#!/usr/bin/env python
# -*- coding: utf-8 -*-

HOST = "localhost"
PORT = 4223
UID = "NNa" # Change XYZ to the UID of your OLED 128x64 Bricklet 2.0

from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_oled_128x64_v2 import BrickletOLED128x64V2

# for ip address
import os
import time

# for wifi
from subprocess import check_output
scanoutput = check_output(["iwlist", "wlan0", "scan"])

try:
    time.sleep(1)

    ipcon = IPConnection()  # Create IP connection

    oled = BrickletOLED128x64V2(UID, ipcon) # Create device object

    ipcon.connect(HOST, PORT) # Connect to brickd
    # Don't use device before ipcon is connected

    # Clear display
    oled.clear_display()

    # get ip address
    ip = os.popen("hostname -I").readline()[:-2]
    # print(ip)

    # # get SSID
    # ssid = ""
    # addlines = False
    # for line in scanoutput.split():
    #
    #     if line.startswith(b'Bit'):
    #         addlines = False
    #         break
    #
    #     if line.startswith(b'ESSID') or addlines is True:
    #         if addlines is False:
    #             ssid = str(line.split(b'"')[1])
    #         else:
    #             ssid += str(line.split(b'"')[0])
    #         addlines = True
    #
    # ssid = ssid.replace("b'", " ")
    # ssid = ssid.replace("'", "")
    # ssid = ssid[1:]

    # Write IP starting from upper left corner of the screen
    # oled.write_line(4, 0, "IP: " + ip)
    # oled.write_line(5, 0, "SSID: " + ssid)
    oled.write_line(0, 0, "Guten Morgen,")
    oled.write_line(1, 0, "du suesse Maus!")
    oled.write_line(3, 0, ":-*")

    ipcon.disconnect()
except:
    pass