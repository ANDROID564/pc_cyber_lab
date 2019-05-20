# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:05:55 2018

@author: admin
"""

import cv2

video_capture = cv2.VideoCapture("2.avi")
while True:
    ret, frame = video_capture.read()
    if ret:
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
video_capture.release()
cv2.destroyAllWindows()