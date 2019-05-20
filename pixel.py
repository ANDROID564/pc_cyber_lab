# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:32:14 2018

@author: admin
"""

import cv2
import numpy as np
 
# read image into matrix.
m =  cv2.imread("b1.jpg")
 
# get image properties.
h,w,bpp = np.shape(m)
 
# print pixel value
y = 1
x = 1
print (m[y][x])