# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 18:54:48 2018

@author: admin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = cv2.imread('b2.jpg')
#cv2.imshow('original image',img)
blurred=cv2.pyrMeanShiftFiltering(img,221,251)
image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
i=-1

_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("no.of contours detected %d"%len(contours))
cv2.drawContours(img,contours,i,(0,0,255),6)
#print('shapes found{0}'.format(len(contours)))
#print("area",cv2.contourArea(contours[i]))

    
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''    
plt.figure('example1')

plt.imshow(img)
plt.show()    
   ''' 

'''
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img=cv2.drawContours(img,[box],0,(0,0,255),12)
        
    cv2.putText(img,"A:{0:2.lf}".format(area),center,
    cv2.POINT_HERSHEY_COMPLEX_SMALL,1.3,(255,0,0),3)

cnt = contours[0]
for cnt in contours:
    
    M=cv2.moments(cnt)
    print(M)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    center=(cx,cy)
    print("center is",center)
    area=cv2.contourArea(cnt)
    print("area is",area)
    perimeter=cv2.arcLength(cnt,True)
    print("perimeter is",perimeter)
    



    '''