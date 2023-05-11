import numpy as np
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
from thresholdOtsu import otsu
from skelet import skeletonize
from dilatation import dilatation
from edgesSegmenation import edges_detection
from incisionPolyline import polyline_detection
import os
import cv2


working = "SA_20220707-193326_incision_crop_0.jpg"  # the working image
test = "SA_20230223-161818_incision_crop_0.jpg"  #  SA_20211012-164802_incision_crop_0.jpg
test2 = "SA_20211012-164802_incision_crop_0.jpg"
test3 = "SA_20230223-124020_incision_crop_0.jpg"
# get all the images
path = "project/images/default"
images_list = os.listdir(path)
false_detected = 0



for image in images_list:


    # load the incision image
    # image = test3
    incision = skimage.io.imread("project/images/default/" + image, as_gray=True)

    img = cv2.imread("project/images/default/" + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    #plt.imshow(threshold, cmap="gray")
    #plt.title('threshold')
    #plt.show()
    """

    skeleton = skeletonize(threshold)
    plt.imshow(skeleton, cmap="gray")
    plt.title('skelet')
    plt.show()
    """

    # binary_0255 = skeleton.astype(np.uint8) * 255
    # Perform Hough Transform to detect lines
    # define the minimum size of detected incision
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=170, minLineLength=(width*0.75), maxLineGap=(width*0.1))

    if lines is not None:

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
        img_with_lines = np.copy(img)
        for line in incisions:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for line in stitches:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert the image to a displayable data type
        img_with_lines_display = cv2.convertScaleAbs(img_with_lines)

        # Display the results
        plt.imshow(cv2.cvtColor(img_with_lines_display, cv2.COLOR_BGR2RGB))
        plt.show()
        a = 0
    else:
        false_detected += 1
        #plt.subplot(211)
        #plt.imshow(img)
        #plt.subplot(212)
        plt.title('threshold')
        #plt.show()
        continue
print(false_detected)
"""
    # thresholding - Otsu
    mask = otsu(incision)

    # skelet
    skeleton = skeletonize(mask)

    # dilatation
    kernel = skimage.morphology.diamond(1)
    bigger_Incision = dilatation(skeleton, kernel)

    plt.figure()
    sub1 = plt.subplot(311)
    plt.imshow(incision, cmap="gray")
    sub1.title.set_text("Incision")
    sub2 = plt.subplot(312)
    plt.imshow(mask, cmap="gray")
    sub2.title.set_text("Otsu")
    sub3 = plt.subplot(313)
    plt.imshow(skeleton, cmap="gray")
    sub3.title.set_text("Skelet")
    plt.show()
    try:
        # incision polyline detection
        polyline = polyline_detection(np.asarray(skeleton), incision, image)
    except Exception as e:
        print(str(e))
"""