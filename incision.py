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


working = "SA_20220707-193326_incision_crop_0.jpg"  # the working image
test = "SA_20230223-161818_incision_crop_0.jpg"  #  SA_20211012-164802_incision_crop_0.jpg
test2 = "SA_20211012-164802_incision_crop_0.jpg"
test3 = "SA_20230223-124020_incision_crop_0.jpg"
# get all the images
path = "project/images/default"
images_list = os.listdir(path)

for image in images_list:
    # load the incision image
    # image = test3
    incision = skimage.io.imread("project/images/default/" + image, as_gray=True)

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
