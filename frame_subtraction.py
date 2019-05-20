o# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:34:27 2018

@author: admin
"""

import numpy as np
import cv2
 
cap = cv2.VideoCapture("2.avi")
 
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
 
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
 
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
 
    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)
 
    key = cv2.waitKey(60)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()