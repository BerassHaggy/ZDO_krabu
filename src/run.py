# Imports
import sys
import numpy as np
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
import os
import cv2
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from anglesComputation import proces_data


# method for detecting stitches based on specific operations
def detect_incision(image, false_detected_incision):

    img = cv2.imread("images/default/" + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # set the scale factor
    scale_percent = 200  # 200% upscaling

    # modify image only if the width is smaller than given threshold
    threshold_width = 200
    if gray.shape[1] < threshold_width:
        # calculate the new dimensions
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled_img = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_img = gray
        out = img

    # thresholding (adaptive)
    threshold = cv2.adaptiveThreshold(upscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=150, minLineLength=(width * 0.75),  # 170 - 46 false
                            maxLineGap=(width * 0.1))

    incisions = []
    if lines is not None:

        # identify incisions and stitches based on their angle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) < 30:  # angle relative to the horizontal axis
                incisions.append(line)
    else:
        scale_percent = 200
        # calculate the new dimensions
        width = int(upscaled_img.shape[1] * scale_percent / 100)
        height = int(upscaled_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled_img = cv2.resize(upscaled_img, dim, interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        threshold = cv2.adaptiveThreshold(upscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17,
                                          2)
        lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=170, minLineLength=(width * 0.75),
                                maxLineGap=(width * 0.1))

        if lines is not None:

            # identify incisions and stitches based on their angle
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if np.abs(angle) < 30:  # angle relative to the horizontal axis
                    incisions.append(line)
        else:
            false_detected_incision += 1

    return incisions, false_detected_incision, out, img


# method for detecting stitches based on specific operations
def detect_stitches(image, false_detected_stitches):

    img = cv2.imread("images/default/" + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 17
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
    out = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    # apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)

    # determine the mean value of the binary image
    mean_value = np.mean(thresh)

    # set Canny parameters based on the mean value
    low_threshold = int(mean_value * 0.5)
    high_threshold = int(mean_value * 1)

    # apply Canny edge detection
    edges = cv2.Canny(thresh, low_threshold, high_threshold)

    dims = edges.shape

    # perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=10, minLineLength=dims[0]*0.3, maxLineGap=dims[0]*0.2)

    # identify stitches based on their angle
    stitches = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) > 60:
                stitches.append(line)
    else:
        false_detected_stitches += 1

    # if stitches are empty, try to adjust the angle
    if len(stitches) < 2:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if np.abs(angle) > 20:
                    stitches.append(line)
        else:
            false_detected_stitches += 1

    return stitches, false_detected_stitches, out


# updating detected coordinates (in a new coordinate system) if image sizes do not correspond
def image_rescale(img_original, img, keypoints):
    # need to compute the ratio between original and incision image
    if img_original.shape == img.shape:
        return keypoints
    else:
        ratio = int(img.shape[0] / img_original.shape[0])  # ratio between two images
        for i in range(0, len(keypoints)):
            keypoints[i][0] = (keypoints[i][0] / ratio).astype(np.int32)
        return keypoints


# method for plotting the detected incisions, stitches, crossings and angles
def draw_detections(incisions, stitches, img_original, image, intersections, intersections_alphas):
    # draw the detected lines on the original image
    img_with_lines = np.copy(img_original)

    # draw the incision
    if incisions is not None:
        for line in incisions:
            # need to compute the ratio between original and incision image
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # draw the stitches
    if stitches is not None and len(stitches[0]) > 0:
        for line in stitches:
            x1, y1, x2, y2 = line[0][0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # convert the image to a displayable data type
    img_with_lines_display = cv2.convertScaleAbs(img_with_lines)

    # display the results
    plt.imshow(cv2.cvtColor(img_with_lines_display, cv2.COLOR_BGR2RGB))

    # draw the crossings
    if len(intersections) > 0:
        for ((xi, yi), alpha) in zip(intersections, intersections_alphas):
            plt.plot(xi, yi, 'o')
            plt.text(xi, yi, '{:2.1f}'.format(alpha), c='green', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1},
                     size='large')

    plt.title("Title: " + image)
    plt.show()
    # skimage.io.imsave(os.path.join("../output_images/one_line_return", image), img_with_lines_display)


# method for averaging all the lines in one class - returning only one line
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

            # draw_detections(keypoints, [], img)

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

    # k - means
    elif keypoints_type == "stitch" and len(keypoints) > 2:
        final_keypoints = k_means(keypoints)
        return final_keypoints

    # only two or one line detected
    else:
        return [keypoints]


# method representing k-means algorithm
def k_means(keypoints):
    k_means_in = list()
    for i in range(len(keypoints)):
        points = keypoints[i][0]
        k_means_in.append(points)
    k_means_in = np.array(k_means_in)
    clusters = dict()

    # identify the classes with corresponding coordinated
    k_values = range(2, len(keypoints))  # Range of k values to try
    silhouette_scores = []  # List to store silhouette scores

    # perform K-means clustering for different values of k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init="auto")
        kmeans.fit(k_means_in)
        labels = kmeans.labels_
        score = silhouette_score(k_means_in, labels)
        silhouette_scores.append(score)

    # find the optimal number of clusters
    best_k = k_values[np.argmax(silhouette_scores)]

    # perform K-means clustering with the best k
    kmeans = KMeans(n_clusters=best_k)
    kmeans.fit(k_means_in)
    labels = kmeans.labels_

    # get the classes and corresponding values
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append([k_means_in[i]])

    # preparing and averaging the detected classes of incisions
    final_keypoints = list()
    for i in range(0, len(clusters.keys())):
        start_points_inc = list()
        end_points_inc = list()
        for j in range(0, len(clusters[i])):
            current_points = clusters[i][0][0]  # x1,y1,x2,y2
            start_part = np.array([current_points[0], current_points[1]])  # x1,y1
            end_part = np.array([current_points[2], current_points[3]])  # x2,y2
            start_points_inc.append(start_part)
            end_points_inc.append(end_part)
        keypoints_average = average_coordinates(np.array([start_part]), np.array([end_part]))
        final_keypoints.append(keypoints_average)
        start_part = list()
        end_part = list()

    return final_keypoints


# method for controlling if there are more lines for one incision which are close
def coordinates_control(keypoints, img, image):
    threshold = img.shape[1]*0.05
    keypoints_final = list()
    banned_lines = list()
    for index1, line1 in enumerate(keypoints):
        for index2 in range(index1+1, len(keypoints)):
            line2 = keypoints[index2]
            line1_midpoint = [(line1[0][0][0] + line1[0][0][2]) / 2, (line1[0][0][1] + line1[0][0][3]) / 2]
            line2_midpoint = [(line2[0][0][0] + line2[0][0][2]) / 2, (line2[0][0][1] + line2[0][0][3]) / 2]
            distance = calculate_distance(line1_midpoint[0], line1_midpoint[1], line2_midpoint[0], line2_midpoint[1])
            if distance <= threshold:
                start_points = [[line1[0][0][0], line1[0][0][1]], [line2[0][0][2], line2[0][0][3]]]
                end_points = [[line1[0][0][2], line1[0][0][3]], [line2[0][0][0], line2[0][0][1]]]
                one_line = average_coordinates(np.array(start_points), np.array(end_points))
                keypoints_final.append(one_line)
                banned_lines.append(line1)
                banned_lines.append(line2)

    for line1 in keypoints:
        if line_in_lines(line1, banned_lines):
            continue
        else:
            keypoints_final.append(line1)
    return keypoints_final


# method for calculating the Euclidian distance
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# support method for detecting if line is already in final lines
def line_in_lines(line, keypoints):
    for points in keypoints:
        if len(points) > 1:
            for point in points:
                if point == line:
                    return True
                    break
        elif np.array_equal(points[0][0], line[0][0]):
            return True
            break
    return False


# method for computing the crossing points and angles and also creating the final dict for json
def compute_crossings_and_angles(image, incision, stitches):
    # image = name of the image
    # incisions = coordinates of the detected incision
    # stitches = coordinates of the detected stitches

    # it is necessary to preprocess the stitches arrays and incisions array
    stitches_in = list()  # as an input for the dedicated method
    incisions_in = list()

    if len(incision) > 0:
        for i in [0, 2]:  # start and end
            coordinates = incision[0][0]
            points = [coordinates[i], coordinates[i+1]]
            incisions_in.append(points)

    incisions_in = [incisions_in]

    # check if any stitches were detected
    if len(stitches[0]) > 0:
        for stitch in stitches:
            coordinates = stitch[0][0]
            line = list()
            for i in [0, 2]:
                points = [coordinates[i], coordinates[i+1]]
                line.append(points)
            stitches_in.append(line)

    # compute the crossings, angles
    intersections, intersections_alphas = proces_data(incisions_in, stitches_in)

    # create the dictionary for json
    information_out = [
        {
                "filename": image,
                "incision_polyline": incision[0][0].tolist(),
                "crossing_positions": str_to_int(intersections, "intersections"),
                "crossing_angles": str_to_int(intersections_alphas, "alphas")
            },
        ]

    intersections_num = str_to_int(intersections, "intersections")
    intersections_alphas_num = str_to_int(intersections_alphas, "alphas")
    return information_out, intersections_num, intersections_alphas_num


# method for writing the information about the input image
def write_to_json(information, filename):
    # information = dictionary type containing incisions, stitches and image filename
    # file = the output json
    with open(filename, "w", encoding='utf-8') as fw:
        json.dump(information, fw, ensure_ascii=False, indent=4)


# method for clearing the json content before the main is called
def clear_json_content(filename):
    with open(filename, "w") as outfile:
        outfile.truncate(0)


# method for converting strings to numbers
def str_to_int(keypoints, type):
    number_keypoints = list()

    if type == "intersections":
        for points in keypoints:
            one_line = list()
            for point in points:
                one_line.append(float(point))
            number_keypoints.append(one_line)

    elif type == "alphas":
        for point in keypoints:
            number_keypoints.append(float(point))
    return number_keypoints


# main method
if __name__ == "__main__":

    """
    # get all the images
    curr_path = "./SP/src/images/default"
    path = os.path.abspath(os.path.join("../../", curr_path))
    print(path)
    images_list = os.listdir(path)
    """
    false_detected_incision = 0
    false_detected_stitches = 0

    # getting the input parameters as an input
    args = sys.argv[1:]  # command line arguments

    output_file = args[0]  # output json filename

    verbose_mode = "-v" in args  # returns True if -v is included - verbose mode activated

    if verbose_mode:
        image_files = args[2:]  # image file names for the detection
    else:
        image_files = args[1:]

    clear_json_content(output_file)
    final_dict = list()

    for image in image_files:
        incisions, false_incision, img_incision, img_original = detect_incision(image, false_detected_incision)
        stitches, false_stitches, img_stitch = detect_stitches(image, false_detected_stitches)
        false_detected_incision = false_incision
        false_detected_stitches = false_stitches
        incisions_out = image_rescale(img_original, img_incision, incisions)
        stitches_out = image_rescale(img_original, img_stitch, stitches)
        incisions = keypoints_postprocessing(incisions_out, img_original, "incision", image)
        stitches = keypoints_postprocessing(stitches_out, img_original, "stitch", image)
        stitches = coordinates_control(stitches, img_original, image)
        information, intersections, intersection_alphas = compute_crossings_and_angles(image, incisions, stitches)

        # add the information for writing to the .json file at the end of the loop
        final_dict.append(information[0])

        # visualise the detected coordinates in the input image
        if verbose_mode:
            draw_detections(incisions, stitches, img_original, image, intersections, intersection_alphas)

    # write the achieved results to json - images that were on the input are included
    write_to_json(final_dict, output_file)
    # print("Incision false detected: ", false_incision)
    # print("Stitches false detected: ", false_stitches)
