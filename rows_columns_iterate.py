# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:23:23 2018

@author: admin
"""

import cv2
import numpy as np

#x = np.random.randint(0,5,(500,500))
img = cv2.imread('b1.jpg',0)
p = img.shape
print (p)
rows,cols = img.shape

for i in range(rows):
    for j in range(cols):
        #k = img[i,j]
        print (img[i,j])