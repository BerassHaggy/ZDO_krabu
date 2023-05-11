import numpy as np
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
# from thresholdOtsu import otsu
# from skelet import skeletonize
# from dilatation import dilatation
# from edgesSegmenation import edges_detection
from incisionPolyline import polyline_detection
import os
import cv2
from pathlib import Path

working = "SA_20220707-193326_incision_crop_0.jpg"  # the working image
test = "SA_20230223-161818_incision_crop_0.jpg"  #  SA_20211012-164802_incision_crop_0.jpg
test2 = "SA_20211012-164802_incision_crop_0.jpg"
test3 = "SA_20230223-124020_incision_crop_0.jpg"
# get all the images
path_images = Path("./images")
images_list = os.listdir(path_images)

for image in images_list:
    # load the incision image
    # image = test3
    incision = skimage.io.imread(os.path.join(path_images, image))

    gray_incision = cv2.cvtColor(incision, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray_incision, 50, 200, apertureSize=3)

    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Identify incisions and stitches based on their angle
    incisions = []
    stitches = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if np.abs(angle) < 30:
            incisions.append(line)
        elif np.abs(angle) > 60:
            stitches.append(line)

    # Draw the detected lines on the original image
    img_with_lines = np.copy(incision)
    for line in incisions:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for line in stitches:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    s# Convert the image to a displayable data type
    img_with_lines_display = cv2.convertScaleAbs(img_with_lines)


    # # thresholding - Otsu
    # mask = otsu(incision)
    #
    # # skelet
    # skeleton = skeletonize(mask)
    #
    # # dilatation
    # kernel = skimage.morphology.diamond(1)
    # bigger_Incision = dilatation(skeleton, kernel)
    #
    # plt.figure()
    # sub1 = plt.subplot(311)
    # plt.imshow(incision, cmap="gray")
    # sub1.title.set_text("Incision")
    # sub2 = plt.subplot(312)
    # plt.imshow(mask, cmap="gray")
    # sub2.title.set_text("Otsu")
    # sub3 = plt.subplot(313)
    # plt.imshow(skeleton, cmap="gray")
    # sub3.title.set_text("Skelet")
    # plt.show()
    # try:
    #     # incision polyline detection
    #     polyline = polyline_detection(np.asarray(skeleton), incision, image)
    # except Exception as e:
    #     print(str(e))

