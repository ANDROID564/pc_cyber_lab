# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:20:27 2018

@author: acer
"""

import numpy as np
import cv2


def nothing(x):    
    pass
         
        #img = cv2.imread("b4.jpg", cv2.IMREAD_GRAYSCALE)
img=cv2.imread("b4.jpg")
    #cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#cv2.namedWindow("Trackbars")
#cv2.createTrackbar("Threshold value", "Trackbars", 128, 255, nothing) 
        
        
        #cv2.resizeWindow('image', 10,10) 
         
while True:   
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([35, 74, 174])
    upper_blue = np.array([76, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #result = cv2.bitwise_and(img, img, mask=mask)
            
            
            
    edges = cv2.Canny(mask, 75, 150)        
            # theta angle and threshold
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, maxLineGap=450)
print (lines) 

j=0
x_center=[0,0]
y_center=[0,0]
x1_low=[]
x2_low=[]
y1_max=[]
y2_max=[]

for line in lines:
    x1, y1, x2, y2 = (line[0])
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    x1_low.append(x1)
    x2_low.append(x2)
    x1_low.extend(x2_low)
    y1_max.append(y1)
    y2_max.append(y2)
    y1_max.extend(y2_max)
x=min(x1_low)
y=max(y1_max)
z=min(y1_max)
z1=max(x1_low)
k1=int((x+z1)/2)
k2=int((y+z)/2)


cv2.line(img, (k1,k2), (k1,k2), (255, 0, 0), 10)
print(k1,k2)
    
cv2.imshow("Edges1", edges)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()