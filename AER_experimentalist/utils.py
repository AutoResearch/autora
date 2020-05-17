import os
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_e_paper_296x128 import BrickletEPaper296x128
from PIL import Image


# Convert PIL image to matching color bool list
def bool_list_from_pil_image(image, width=296, height=128, color=(0, 0, 0)):
    image_data = image.load()
    pixels = []

    for row in range(height):
        for column in range(width):
            pixel = image_data[column, row]
            value = (pixel[0] == color[0]) and (pixel[1] == color[1]) and (pixel[2] == color[2])
            pixels.append(value)

    return pixels

def init_epaper():
    HOST = "localhost"
    PORT = 4223
    UID = "Lxb"  # Change XYZ to the UID of your E-Paper 296x128 Bricklet

    WIDTH = 296  # Columns
    HEIGHT = 128  # Rows

    try:
        ipcon = IPConnection()  # Create IP connection
        epaper = BrickletEPaper296x128(UID, ipcon)  # Create device object

        ipcon.connect(HOST, PORT)  # Connect to brickd
        # Don't use device before ipcon is connected

        # Download example image here:
        # https://raw.githubusercontent.com/Tinkerforge/e-paper-296x128-bricklet/master/software/examples/tf_red.png
        image = Image.open('/home/pi/PycharmProjects/AER_experimentalist/images/AER_experimentalist_active.png')
        # image = Image.open('/home/pi/PycharmProjects/AER_experimentalist/images/pi_epaper.png')

        # Get black/white pixels from image and write them to the Bricklet buffer
        pixels_bw = bool_list_from_pil_image(image, WIDTH, HEIGHT, (0xFF, 0xFF, 0xFF))
        epaper.write_black_white(0, 0, WIDTH - 1, HEIGHT - 1, pixels_bw)

        # Get red pixels from image and write them to the Bricklet buffer
        pixels_red = bool_list_from_pil_image(image, WIDTH, HEIGHT, (0xFF, 0, 0))
        epaper.write_color(0, 0, WIDTH - 1, HEIGHT - 1, pixels_red)

        # Draw buffered values to the display
        epaper.draw()

        ipcon.disconnect()
    except:
        pass


def trial_to_list(trial=None, IVList = None, DVList = None):

    messages = list()

    if trial is not None:
        messages.append("--- Step " + str(trial) + " ---")

    if IVList is not None:
        for IV in IVList:
            messages.append("IV " + str(IV[0]) + " = " + str(IV[1]))

    if DVList is not None:
        messages.append("--- Measurement: ")
        for DV in DVList:
            messages.append("DV " + str(DV[0]) + " = " + str(round(DV[1],4)))

    return messages

def get_experiment_files(path):

    experiment_files = list()

    for file in os.listdir(path):
        if file.endswith(".exp"):
            experiment_files.append(str(file)) # os.path.join(path, file)

    return experiment_files
