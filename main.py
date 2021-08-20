"""
    Test bench for interpoaltion class
    Arthor: Zhenhuan(Steven) Sun
"""

# from interpolation module import interpolation class
from interpolation import interpolation
import cv2
import numpy as np

interp = interpolation(image_path="./Image/man.png", scale=10)

# resize image using different interpolation method
resized_image_nn = interp.nearest_neighbour()
resized_image_bilinear = interp.bilinear()
resized_image_bicubic = interp.bicubic()

# stack image matrix horizontally in order to show 3 images in one window
image = np.hstack((resized_image_nn, resized_image_bilinear, resized_image_bicubic))

cv2.imshow("Nearest Neighbour Interpolation", resized_image_nn)
cv2.imshow("Bilinear Interpolation", resized_image_bilinear)
cv2.imshow("Bicubic Interpolation", resized_image_bicubic)
cv2.imshow("Interpolation comparision", image)
cv2.waitKey(0)
cv2.destroyAllWindows()