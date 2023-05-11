import matplotlib.pyplot as plt
from pathlib import Path
import os
import skimage.io
import cv2

path_images = Path("./images")
images_list = os.listdir(path_images)

for image in images_list:
    # load the incision image
    # image = test3
    incision = skimage.io.imread(os.path.join(path_images, image))
    plt.imshow(incision)
    plt.show()
