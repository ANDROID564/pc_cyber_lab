
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:26:36 2018

@author: acer
function made and calle dfrom previous test1 mask value
"""

import numpy as np
import cv2
from test_frames_creation2 import mask

class analyse_frame:
    def frame_new1(filename):
        filename1=filename
        h,w,bpp = np.shape(filename1)
            #print(np.shape(m))
        y = 1
        x = 1
        row=h
        col=w
        print(filename1[0][0][0])
            
           
        if(filename1[0][0][0]>0):
                
            #filename=cv2.imread(filename)
            img=mask.canny(filename)
            
            
            #cv2.imshow("img1",img)
            #img=mask.canny(cv2.imread(filename)) 
            #cv2.imshow("Image2", img)
            #img = cv2.imread("b5.jpg")
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img, 75, 150)
            
                
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, maxLineGap=450)
            #print (lines) 
                
            j=0
            x_center=[0,0]
            y_center=[0,0]
            m_center=[0,0]
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
            m_center=(k1,k2)
            print(m_center)
            
            cv2.line(img, (k1,k2), (k1,k2), (255, 0, 0), 10)
            #print(k1,k2)
            '''
            cv2.imshow("Edges1", edges)
            cv2.imshow("Image", img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #return img
            '''
            return m_center
        #(frame_new1(cv2.imread('b4.jpg')))        
        else:
            return 0