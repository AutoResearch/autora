#!/usr/bin/env python
# -*- coding: utf-8 -*-

HOST = "localhost"
PORT = 4223
UID = "Lxb"  # Change XYZ to the UID of your E-Paper 296x128 Bricklet

WIDTH = 296  # Columns
HEIGHT = 128  # Rows

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

def update_epaper():

    ipcon = IPConnection()  # Create IP connection
    epaper = BrickletEPaper296x128(UID, ipcon)  # Create device object

    ipcon.connect(HOST, PORT)  # Connect to brickd
    # Don't use device before ipcon is connected

    # Download example image here:
    # https://raw.githubusercontent.com/Tinkerforge/e-paper-296x128-bricklet/master/software/examples/tf_red.png
    image = Image.open("/home/pi/PycharmProjects/AER_experimentalist/images/AER_experimentalist_offline.png")
    #image = Image.open('tf_red.png')

    # Get black/white pixels from image and write them to the Bricklet buffer
    pixels_bw = bool_list_from_pil_image(image, WIDTH, HEIGHT, (0xFF, 0xFF, 0xFF))
    epaper.write_black_white(0, 0, WIDTH - 1, HEIGHT - 1, pixels_bw)

    # Get red pixels from image and write them to the Bricklet buffer
    pixels_red = bool_list_from_pil_image(image, WIDTH, HEIGHT, (0xFF, 0, 0))
    epaper.write_color(0, 0, WIDTH - 1, HEIGHT - 1, pixels_red)

    # Draw buffered values to the display
    epaper.draw()

    ipcon.disconnect()

if __name__ == "__main__":
    update_epaper()