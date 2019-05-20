import numpy as np
import cv2
 
img = cv2.imread("b7.jpg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 75, 150)
 
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, maxLineGap=450)
#print (lines) 

j=0
x_center=[0,0]
y_center=[0,0]
x1_low=[]
x2_max=[]
y1_low=[]
y2_max=[]
x1y1=[]
x2y2=[]
distance_x=[]
distance_y=[]
distance_u=[]

'''
for i in range(5):
    x_low[i]=1
'''

for line in lines:
    x1, y1, x2, y2 = (line[0])
    k1=cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    x=x2-x1
    y=y2-y1
            
    if(y2==y1):
        dist=int(x*x+y*y)
        #dist=int((x2-x1)+(y2-y1))
        print("distance x-axis",line,dist)
        distance_x.append(dist)
    elif(x2==x1):
        dist=int(x*x+y*y)
        #dist=int((x2-x1)+(y2-y1))
        print("distance y-axis",line,dist)
        distance_y.append(dist)
    '''
    else:
        dist=int(x*x+y*y)
        #dist=int((x2-x1)+(y2-y1))
        print("distance u-axis",line,dist)
        distance_u.append(dist)
    '''

           
    
    t1=[(x1,y1)]     
    x1y1.append(t1)
    #print("x1y1******************",x1y1)
    t2=[(x2,y2)]     
    x2y2.append(t2)
    #print("x2y2******************",x2y2)
    x1_low.append(x1)
    x2_max.append(x2)
    x1_low.extend(x2_max)
    y1_low.append(y1)
    y2_max.append(y2)
    y1_low.extend(y2_max)
    #print("#####",y1_low)
    
    k1=len(x1y1)
    #print("length x1y1",k1)
    k2=len(x2y2)
    #print("length x2y2",k2)
    '''
    if(y1==y2):
        j=1
        for i in range(len(x1_low)):
            if(x1_low[i]==x1_low[j] ):
                print("horizontal:",(x1_low[i],y1)
             #   j=j+1
                
            elif( y2_max[i]==y2_max[j]):
                print("horizontal:",(x1_low[i],y1)
                j=j+1
    
    elif(x1==x2):
        j=1
        for i in range(len(y1_low)):
            if(y1_low[i]==y1_low[j] or y2_max[i]==y2_max[j]):
                print("vertical:")
                j=j+1
                
     '''       
                
        
    
    j=1
    '''
    for i in range(k1):
       if(y1==y2):
           dist=int((x2-x1)+(y2-y1))
           print("distance",dist)
           break
    '''
    
print("distances of all the x lines",distance_x)
print("distances of all the y lines",distance_y)
#print("distances of all the u lines",distance_u)
        
    
x=min(x1_low)
y=max(y1_low)
z=min(y1_low)
z1=max(x1_low)
k1=int((x+z1)/2)
k2=int((y+z)/2)
print ("min_x1",min(x1_low))
print ("min_y1",min(y1_low))    
print ("max_x2",max(x1_low))
print ("max_y2",max(y1_low))
    #print(x1)
'''    if(i==7 or i==10):    
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
    x=int((x2+x1)/2)
    y=int((y2+y1)/2)
    x_center[j]=x
    y_center[j]=y
    x_low[]
    print("line_no.",i,line,"x-center",x_center[j],"y-center",y_center[j])
    j=j+1
    i=i+1'''
#print (x_low)
'''
x=int((x_center[0]+x_center[1])/2 )
y=int((y_center[0]+y_center[1])/2 )
print(x,y)

'''
            
        



#cv2.line(img, (303,239), (428,243), (255, 0, 0), 4)
cv2.line(img, (k1,k2), (k1,k2), (255, 0, 0), 10)
print(k1,k2)
    
cv2.imshow("Edges", edges)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()