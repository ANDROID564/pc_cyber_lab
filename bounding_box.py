# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:10:34 2018

@author: admin
"""

import cv2
import numpy as np
from PIL import Image
import numpy as np
import cv2 as cv
img = cv.imread('b1.jpg',0)
ret,thresh = cv.threshold(img,127,255,0)
_,contours,hierarchy = cv.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('shapes found{0}'.format(len(contours)))

cnt = contours[0]
M = cv.moments(cnt)
print( M )#print(gray)
cx=int(M['m10']/M['m00'])
cy=int(M['m01']/M['m00'])
center=(cx,cy)
print("center is",center)

#cv2.imwrite( 't4.jpg', roi_copy)
#cv2.imshow('img',img)
'''
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
    
'''