import numpy as np
import cv2
 
class mask:
    def canny(img):
        
        def nothing(x):    
            pass
         
        #img = cv2.imread("b4.jpg", cv2.IMREAD_GRAYSCALE)
        #img=cv2.imread(image)
        #cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
        cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        #cv2.namedWindow("Trackbars")
        #cv2.createTrackbar("Threshold value", "Trackbars", 128, 255, nothing) 
        h,w,bpp = np.shape(img)
#print(np.shape(m))
        y = 1
        x = 1
        row=h
        col=w
        print(img[0][0][0])
          
        
        
        #cv2.resizeWindow('image', 10,10) 
        if(img[0][0][0]>0):
                
             
            while True:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                lower_blue = np.array([35, 74, 174])
                upper_blue = np.array([76, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                result = cv2.bitwise_and(img, img, mask=mask)
                
                
                
                edges = cv2.Canny(mask, 75, 150)        
                # theta angle and threshold
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100)
                
                #print (lines) 
                for line in lines:
                    
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                   
                cv2.imshow("Image", img)
                cv2.imshow("mask", mask)
                cv2.imshow("result", result)
                cv2.imshow("Edges", edges)
                
                key = cv2.waitKey(100)
                if key == 27:
                    break
             
                #cv2.destroyAllWindows()
                
                return mask
        else:
            return 0
                
                
             
    
            
        