# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:30:56 2018

@author: acer
"""

import numpy as np
import cv2
 
class mask:
    def canny(img):
        
        def nothing(x):    
            pass
       
        cv2.resize(img, (0,0), fx=0.5, fy=0.5)
  
         
        while True:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                                   
            lower_blue = np.array([55, 74, 174])
            upper_blue = np.array([76, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
         
            ret,thresh_img = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
            arr=np.asarray(thresh_img)
            tot=arr.sum(-1)
            mn  = arr.mean(-1)
            tot1 = arr.sum(0).sum(0)
          
            if(tot1>20000):
                result = cv2.bitwise_and(img, img, mask=mask)
                
           
                key = cv2.waitKey(100)
                if key == 27:
                    break
                    
           
                cv2.destroyAllWindows()
            
                return mask
            else:
                return 'NULL'
            
         

        
    