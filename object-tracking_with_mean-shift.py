# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:30:51 2018

@author: admin
"""

import numpy as np
import cv2 
video = cv2.VideoCapture("2.avi")
 
_, first_frame = video.read()
x = 354
y = 252
width = 455 - x
height = 395 - y
roi = first_frame[y: y + height, x: x + width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
 
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
 
while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
 
    _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
 
    key = cv2.waitKey(100)
    if key == 27:
        break
 
video.release()
cv2.destroyAllWindows()