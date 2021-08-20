import numpy as np
import cv2

image = cv2.imread('./Image/img_example_lr.png')
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

class interpolation():
    def __init__(self, interpolation="bicubic", scale=1):
        self.interpolation = interpolation
        self.scale = scale

    
