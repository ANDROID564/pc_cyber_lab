# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:42:38 2018

@author: admin
"""

import numpy as np
import os
import cv2
img = cv2.imread("b1.jpg")
roi = img[246: 369, 388: 485]
x = 408
y = 256
width = 465 - x
height = 349 - y
cv2.imshow('img',roi)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
 
cap = cv2.VideoCapture('2.avi')
 
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
i=0     
currentFrame=0
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
 
    ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)
    print("frame no.",str(i),ret)
    i+=1
    
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    
    cv2.imshow("mask", mask)
    cv2.imshow("Frame", frame)
 
    key = cv2.waitKey(30)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
