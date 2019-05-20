# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:01:04 2018

@author: admin
"""

import Image, numpy
import cv2
import numpy as np
numpy.asarray(Image.open('b1.jpg').convert('L'))
'''
img = cv2.imread('b1.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print (M)

''' 
    