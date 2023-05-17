import matplotlib.pyplot as plt
from pathlib import Path
import os
import skimage.io
import cv2
import math
import numpy as np


def count_angles(stitches_list: list, incision):
    res_angles = list()

    incision_start = incision[0] # tuple
    incision_end = incision[1] # tuple

    for stitch in stitches_list: # stitches_list = [[(start), (end)], [(start), (end)],...[]]
        stitch_start = stitch[0]
        stitch_end = stitch[1]

        angle = angle_between_lines(stitch_start, stitch_end, incision_start, incision_end)
        res_angles.append(angle)

def angle_between_lines(p1_start, p1_end, p2_start, p2_end):
    # Compute the slopes of the lines
    dx1 = p1_end[0] - p1_start[0]
    dy1 = p1_end[1] - p1_start[1]
    dx2 = p2_end[0] - p2_start[0]
    dy2 = p2_end[1] - p2_start[1]

    # Calculate the angles using arctan2
    theta1 = np.arctan2(dy1, dx1)
    theta2 = np.arctan2(dy2, dx2)

    # Compute the angle between the lines
    angle = abs(theta2 - theta1)

    # Normalize the angle to the range [0, pi/2]
    if angle > math.pi / 2:
        angle = math.pi - angle

    # Convert angle to degrees
    angle_degrees = math.degrees(angle)

    return angle_degrees


x_1_stitch = [30, 30]
y_1_stitch = [10, 50]

x_2_stitch = [26, 24]
y_2_stitch = [10, 50]

x_3_stitch = [24, 22]
y_3_stitch = [10, 50]

x_incision = [22, 35]
y_incision = [25, 40]

plt.plot(x_1_stitch, y_1_stitch)
plt.plot(x_2_stitch, y_2_stitch)
plt.plot(x_3_stitch, y_3_stitch)
plt.plot(x_incision, y_incision)
plt.axvline(x=0, label="x=0")
plt.axhline(y=0, label="y=0")
plt.show()

stitch_start = [24, 10]
stitch_end = [22, 50]
incision_start = [22, 25]
incision_end = [35, 40]

angle = angle_between_lines(stitch_start, stitch_end, incision_start, incision_end)
print(angle)
