# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:51:11 2018

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:12:22 2018

@author: admin
"""
from video_pixels import video
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
cap = cv2.VideoCapture('2.avi')

hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
 
roi = cv2.imread("person.jpg")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
hue, saturation, value = cv2.split(hsv_roi)
 
 
# Histogram ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
 
# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
 
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(original_image, mask)
 
cv2.imshow("Mask", mask)
cv2.imshow("Original image", original_image)
cv2.imshow("Result", result)
cv2.imshow("Roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()