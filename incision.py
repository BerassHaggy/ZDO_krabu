import numpy as np
import scipy
from scipy import ndimage
import scipy.signal
import scipy.misc
import skimage.data
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
from skimage.feature import canny
from scipy import ndimage
from skimage.segmentation import active_contour
from skimage.filters import sobel


incision = skimage.io.imread("project/images/default/SA_20230223-082339_incision_crop_2.jpg", as_gray=True)
# print the image
plt.imshow(incision, cmap="gray")
#plt.show()

# thresholding - Otsu
threshold = filters.threshold_otsu(incision)
mask = incision < threshold
plt.imshow(mask, cmap="gray")


# edges based segmentation
edges = canny(incision)
plt.imshow(edges, cmap="gray")
#plt.show()
incision_edges = ndimage.binary_fill_holes(edges)
plt.imshow(incision_edges, cmap="gray")
#plt.show()

# erosion
kernel_big = skimage.morphology.diamond(1)
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
new = skimage.morphology.binary_erosion(edges, kernel)
new2 = skimage.morphology.binary_opening(edges, kernel_big)
#plt.imshow(new2)
#plt.show()

# skelet
skeleton = skimage.morphology.skeletonize(mask)

# dilatation
kernel2 = np.array([[1, 0], [0, 0]])
new3 = skimage.morphology.binary_dilation(skeleton, kernel_big)

"""
# sobel
new4 = elevation_map = sobel(incision)
markers = np.zeros_like(new4)
markers[incision < 100] = 1
markers[incision > 110] = 2
segmentation = skimage.segmentation.watershed(new4, markers)
"""

plt.figure()
plt.subplot(411)
plt.imshow(skeleton, cmap="gray")
plt.subplot(412)
plt.imshow(incision, cmap="gray")
plt.subplot(413)
plt.imshow(new3, cmap="gray")
plt.show()


