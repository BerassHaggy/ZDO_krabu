# Imports
import numpy as np
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
import os
import cv2
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Imports from own modules
from thresholdOtsu import otsu
from skelet import skeletonize
from dilatation import dilatation
from edgesSegmenation import edges_detection
from incisionPolyline import polyline_detection


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

    return incisions, false_detected_incision, out, img


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
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) > 30:
                stitches.append(line)
    else:
        false_detected_stitches += 1

    return stitches, false_detected_stitches, out


def image_rescale(img_original, img, keypoints):
    # need to compute the ratio between original and incision image
    if img_original.shape == img.shape:
        return keypoints
    else:
        ratio = int(img.shape[0] / img_original.shape[0])  # ratio between two images
        for i in range(0, len(keypoints)):
            keypoints[i][0] = (keypoints[i][0] / ratio).astype(np.int32)
        return keypoints


def draw_detections(incisions, stitches, img_original):
    # Draw the detected lines on the original image
    img_with_lines = np.copy(img_original)
    for line in incisions:
        # need to compute the ratio between original and incision image
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for line in stitches:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # convert the image to a displayable data type
    img_with_lines_display = cv2.convertScaleAbs(img_with_lines)

    # display the results
    plt.imshow(cv2.cvtColor(img_with_lines_display, cv2.COLOR_BGR2RGB))
    plt.show()


def average_coordinates(start_points, end_points):
    keypoints_out = list()
    # controlling if any incisions and stitches were detected
    if len(start_points) != 0 and len(end_points) != 0:
        start_x = np.mean(start_points[:, 0]).astype(np.int32)
        start_y = np.mean(start_points[:, 1]).astype(np.int32)
        end_x = np.mean(end_points[:, 0]).astype(np.int32)
        end_y = np.mean(end_points[:, 1]).astype(np.int32)

        # returning the proper format
        keypoints_out.append(np.array([[start_x, start_y, end_x, end_y]]))

    return keypoints_out


def keypoints_postprocessing(keypoints, img, keypoints_type, image):
    if keypoints_type == "incision":
        threshold_band = img.shape[0]*0.05
        start_points = list()  # for start points
        end_points = list()  # for end points

        # for the final output
        start_points_out = list()
        end_points_out = list()
        far_keypoints = list()
        if len(keypoints) > 1:
            for line_part in [0, 2]:  # starts then ends
                for i in range(0, len(keypoints)):  # getting start/end points (x,y)
                    current_points = keypoints[i][0]  # x1,y1,x2,y2
                    curr_part = np.array([current_points[line_part], current_points[line_part+1]])

                    # store the corresponding coordinates
                    if line_part == 0:
                        start_points.append(curr_part)
                    else:
                        end_points.append(curr_part)

            # making the average of the detected coordinates (x,y)
            keypoints_out = average_coordinates(np.array(start_points), np.array(end_points))

            # computing the reference points
            x_start = keypoints_out[0][0][0]
            y_start = keypoints_out[0][0][1]
            x_end = keypoints_out[0][0][2]
            y_end = keypoints_out[0][0][3]

            #draw_detections(keypoints, [], img)

            for line_part in [0, 2]:  # starts then ends
                for i in range(0, len(keypoints)):  # getting start/end points (x,y)
                    current_points = keypoints[i][0]  # x1,y1,x2,y2
                    curr_part = np.array([current_points[line_part], current_points[line_part+1]])  # x1,y1 or x2,y2
                    y_current = curr_part[1]  # y real

                    if line_part == 0:  # detecting the start points
                        if np.abs(y_start - y_current) <= threshold_band:
                            start_points_out.append(curr_part)
                        else:
                            start_points_out.append(curr_part)
                    else:
                        if np.abs(y_end - y_current) <= threshold_band:
                            end_points_out.append(curr_part)
                        else:
                            end_points_out.append(curr_part)

            # compute the final coordinates
            keypoints_out = average_coordinates(np.array(start_points_out), np.array(end_points_out))
            if len(keypoints_out) == 0:
                print(image)
            return keypoints_out
        else:
            return keypoints  # returning the (x1,y1) (x2,y2)
    elif keypoints_type == "stitch":
        a = 0


def angle_between_lines(p1_start, p1_end, p2_start, p2_end):
    '''
    Could be used for counting angles between incision and stitches
    :param p1_start:
    :param p1_end:
    :param p2_start:
    :param p2_end:
    :return:
    '''
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


if __name__ == "__main__":
    working = "SA_20220707-193326_incision_crop_0.jpg"  # the working image
    test = "SA_20230223-161818_incision_crop_0.jpg"  #  SA_20211012-164802_incision_crop_0.jpg
    test2 = "SA_20211012-164802_incision_crop_0.jpg"
    test3 = "SA_20221014-114727_incision_crop_0.jpg"
    # get all the images
    path = "project/images/default"
    images_list = os.listdir(path)
    false_detected_incision = 0
    false_detected_stitches = 0

    for image in images_list:
        incisions, false_incision, img_incision, img_original = detect_incision(image, false_detected_incision)
        stitches, false_stitches, img_stitch = detect_stitches(image, false_detected_stitches)
        false_detected_incision = false_incision
        false_detected_stitches = false_stitches
        incisions_out = image_rescale(img_original, img_incision, incisions)
        stitches_out = image_rescale(img_original, img_stitch, stitches)
        incisions = keypoints_postprocessing(incisions, img_incision, "incision", image)
        draw_detections(incisions, stitches_out, img_original)
    print("Incision false detected: ", false_incision)
    print("Stitches false detected: ", false_stitches)
