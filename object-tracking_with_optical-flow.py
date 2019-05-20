# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:39:35 2018

@author: admin
"""

import numpy as np
import cv2
 
cap = cv2.VideoCapture('2.avi')
 
# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
# Lucas kanade params
lk_params = dict(winSize = (20, 20),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
  
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
 
point_selected = False
point = ()
old_points = np.array([[]])
print("selected_point",old_points[0])

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        #print("selected_point",point)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        print(new_points)
        old_points = new_points
 
        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
 
 
 
    #first_level=cv2.pyrDown(frame)
    #second_level=cv2.pyrDown(frame)
    #print("old_points",old_points)    
    #print("last_point",new_points)  

    
    cv2.imshow("Frame", frame)
    #cv2.imshow("first_level",first_level)
    #cv2.imshow("second_level",second_level)

 
    key = cv2.waitKey(100)
    
    if key == 27:
        break
    
#print("selected_point",point)
print("selected_point",point[0])
#print("old_points",old_points)    
print("last_point",new_points) 
#distance=point-new_points
#print("distance",distance)
cap.release()
cv2.destroyAllWindows()
