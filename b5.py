# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:16:52 2018

@author: admin
"""

import numpy as np
import cv2

im = cv2.imread('b1.jpg')
im[im == 255] = 1
im[im == 0] = 255
im[im == 1] = 0
im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(im2,127,255,0)
_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i in range(0, len(contours)):
    if (i % 2 == 0):
       cnt = contours[i]
       #mask = np.zeros(im2.shape,np.uint8)
       #cv2.drawContours(mask,[cnt],0,255,-1)
       x,y,w,h = cv2.boundingRect(cnt)
       cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
       cv2.imshow('Features', im)
       cv2.imwrite(str(i)+'.png', im)

cv2.destroyAllWindows()