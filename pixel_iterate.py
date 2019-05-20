# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:33:01 2018

@author: admin
"""

import cv2
import numpy as np
 
# read image into matrix.
m =  cv2.imread("b1.jpg")
 
# get image properties.
h,w,bpp = np.shape(m)
 
# iterate over the entire image.
for py in range(0,h):
    print(py,"next row started")
    for px in range(0,w):
        print (m[py][px])
        
        
        
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
