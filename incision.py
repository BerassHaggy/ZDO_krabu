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


def detect_incision(image, false_detected_incision):
    incision = skimage.io.imread("project/images/default/" + image, as_gray=True)

    img = cv2.imread("project/images/default/" + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the scale factor
    scale_percent = 200  # 200% upscaling

    # modify image only if the width is smaller than given threshold
    threshold_width = 200
    if gray.shape[1] < threshold_width:
        # Calculate the new dimensions
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled_img = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_img = gray
        out = img

    # Thresholding (adaptive)
    threshold = cv2.adaptiveThreshold(upscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    plt.subplot(211)
    plt.imshow(gray, cmap="gray")
    plt.subplot(212)
    plt.imshow(upscaled_img, cmap="gray")

    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=150, minLineLength=(width * 0.75),  # 170 - 46 false
                            maxLineGap=(width * 0.1))
    incisions = []
    if lines is not None:

        # Identify incisions and stitches based on their angle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) < 30:  # angle relative to the horizontal axis
                incisions.append(line)
    else:
        scale_percent = 200
        # Calculate the new dimensions
        width = int(upscaled_img.shape[1] * scale_percent / 100)
        height = int(upscaled_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled_img = cv2.resize(upscaled_img, dim, interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        threshold = cv2.adaptiveThreshold(upscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17,
                                          2)
        lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=170, minLineLength=(width * 0.75),
                                # 170 - 46 false
                                maxLineGap=(width * 0.1))
        if lines is not None:

            # Identify incisions and stitches based on their angle
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if np.abs(angle) < 30:  # angle relative to the horizontal axis
                    incisions.append(line)
        else:
            false_detected_incision += 1
            #plt.imshow(upscaled_img, cmap="gray")
            #plt.show()
            a = 0

    return incisions, false_detected_incision, out


def detect_stitches(image, false_detected_stitches):

    img = cv2.imread("project/images/default/" + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 17
    gray= cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
    out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)


    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)

    # Determine the mean value of the binary image
    mean_value = np.mean(thresh)

    # Set Canny parameters based on the mean value
    low_threshold = int(mean_value * 0.5)
    high_threshold = int(mean_value * 1)

    # Apply Canny edge detection
    edges = cv2.Canny(thresh, low_threshold, high_threshold)

    dims = edges.shape

    plt.subplot(211)
    plt.imshow(thresh, cmap="gray")
    plt.subplot(212)
    plt.imshow(edges, cmap="gray")
    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=0.5, theta=np.pi / 180, threshold=10, minLineLength=dims[0]*0.1, maxLineGap=20)

    # Identify stitches based on their angle
    stitches = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if np.abs(angle) > 30:
            stitches.append(line)

    return stitches, false_detected_stitches, out


def draw_detections(incisions, stitches, img):
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


if __name__ == "__main__":
    working = "SA_20220707-193326_incision_crop_0.jpg"  # the working image
    test = "SA_20230223-161818_incision_crop_0.jpg"  #  SA_20211012-164802_incision_crop_0.jpg
    test2 = "SA_20211012-164802_incision_crop_0.jpg"
    test3 = "SA_20230223-124020_incision_crop_0.jpg"
    # get all the images
    path = "project/images/default"
    images_list = os.listdir(path)
    false_detected_incision = 0
    false_detected_stitches = 0

    for image in images_list:
        incisions, false_incision, img = detect_incision(image, false_detected_incision)
        stitches, false_stitches, img = detect_stitches(image, false_detected_stitches)
        false_detected_incision = false_incision
        false_detected_stitches = false_stitches
        draw_detections(incisions, stitches, img)
    print("Incision false detected: ", false_incision)
    print("Stitches false detected: ", false_stitches)
