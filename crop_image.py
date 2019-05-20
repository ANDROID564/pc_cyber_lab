# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:45:10 2018

@author: admin
"""

import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    im = cv2.imread("b1.jpg")
     
    # Select ROI
    #r = cv2.selectROI(im)
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", im, fromCenter, showCrosshair)
     
    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)