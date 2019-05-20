
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:26:36 2018

@author: acer
function made and called from previous test1 mask value
"""

import numpy as np
import cv2
from test_frames_main2 import mask

class analyse_frame:
    def frame_new1(filename):
        filename1=filename
        #filename is the frame from the main file
        img=mask.canny(filename)
#      
        while(True):    
            if(img=='NULL'):
                tot1=0
                break
            else:
                ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                arr=np.asarray(thresh_img)
                tot=arr.sum(-1)
                mn  = arr.mean(-1)
                tot1 = arr.sum(0).sum(0)
                break
    
        img1=img
        while(True):   
            if np.any((img1=='NULL')==False and tot1>0):
                edges = cv2.Canny(img, 75, 150)
                
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, maxLineGap=450)
                  
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
       
                
                cv2.line(img, (k1,k2), (k1,k2), (255, 0, 0), 10)
            
                return m_center
           
            else:
          
                return 'NULL'
       