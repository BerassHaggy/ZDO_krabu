import matplotlib.pyplot as plt
from pathlib import Path
import os
import skimage.io
import cv2
import math
import numpy as np


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


x_1 = [30, 30]
y_1 = [10, 50]

x_2 = [25, 35]
y_2 = [20, 35]

plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
plt.show()

p1_start = [30, 10]
p1_end = [30, 50]
p2_start = [25, 20]
p2_end = [35, 35]

angle = angle_between_lines(p1_start, p1_end, p2_start, p2_end)
print()
