# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:12:22 2018

@author: admin
"""
from main1 import original_video
from video_pixels import video
import numpy as np
import cv2
from matplotlib import pyplot as plt

#here answer of video_pixels is printed
print(video.detect(cv2.VideoCapture('2.avi'))) 

#here we will get the result of video file from main1.py
cap=original_video.detect('2.avi')


#original_image = cv2.imread("b3.jpg")
#cap = cv2.VideoCapture("2.avi")




#hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
#hsv_original = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
  
roi = cv2.imread("person.jpg")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
hue, saturation, value = cv2.split(hsv_roi)
 
while True:
    ret,frame=cap.read()
    hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hue, saturation, value = cv2.split(hsv_roi)
     
# Histogram ROI
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
 
# Filtering remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
 
    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(frame, mask)
 
    cv2.imshow("Mask", mask)
    cv2.imshow("Original video", frame)
    cv2.imshow("Result", result)
    cv2.imshow("Roi", roi)
    
    key = cv2.waitKey(60)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()