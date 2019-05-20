import numpy as np
import cv2
 
img = cv2.imread("b5.jpg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 75, 150)
 
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, maxLineGap=450)
print (lines) 
i=1
j=0
x_center=[0,0]
y_center=[0,0]
for line in lines:
    x1, y1, x2, y2 = (line[0])
    if(i==7 or i==10):    
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
        x=int((x2+x1)/2)
        y=int((y2+y1)/2)
        x_center[j]=x
        y_center[j]=y
        j=j+1
        print("line_no.",i,line,"x-center",x_center[0],"y-center",y_center[0])
    i=i+1
 
#cv2.line(img, (300, 316), (346, 316), (255, 0, 0), 2)
    
cv2.imshow("Edges", edges)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()