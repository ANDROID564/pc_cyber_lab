# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:37:23 2018

@author: admin
"""

import numpy as np
import cv2
 
cap = cv2.VideoCapture("2.avi")
 
subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
 
while True:
    _, frame = cap.read()
 
    mask = subtractor.apply(frame)
 
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
 
    key = cv2.waitKey(60)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()