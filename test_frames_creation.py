# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:11:14 2018

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:43:10 2018

@author: admin
"""

#files:frames
#img3:array of file_path+folder_name
#img1:frames in pixel format
#img2:apppending folder with file
import os
clear = lambda: os.system('cls')
clear()

import cv2
import numpy as np
import os
from test_frames_creation1 import analyse_frame
from PIL import Image
import matplotlib.pyplot as plt
 
from os.path import isfile, join
 
#def convert_frames_to_video(pathIn,pathOut,fps):
def convert_frames_to_video(pathIn):

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    #print(files)
    #filename=[]
    j=0
    img1=[]
    img3=[]
    img1.append(files)
    #print(img1)
    for i in range(len(files)):
        #filename=pathIn + files[i]
        #print(files[i])
        img2=pathIn + files[i]
        img3.append(img2)
        #print(img3)
        #j=j+1
        #print(filename)
        #cv2.imshow(img3)
        #reading each files
        img1 = cv2.imread(img2)
        #img1.append(files)
        #print(cv2.imread(img1))
        
       
        #print(i,img1)
        #print(cv2.imread(filename))
        '''
        img1=analyse_frame.frame_new1(cv2.imread(img1))
        break
        #img1=mask.canny(cv2.imread(filename[i]))
        
        height, width, layers = img1.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img1)
        
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
        '''
    i=0    
    j=0
    image_array=[]
    #print(img3)
    print(len(img3))
    for i in range(len(img3)):
        #print(i)
        #j=78
        #if(i==j and j<121):
        if(i==j):
        #if(i==j and j<63):
            print(i,(img3[i]))
            #print((img1[i]))
            #print((files[i]))
            #k=Image.open('photo.jpg')
            #images = np.array(Image.open('img3[i]'))
            #images= (Image.open(img3[i]))
            k=img3[i]
            #print(images)
            print(j,k)
            #img=cv2.imread(images)
            #img=cv2.imread('z1video_frames2/frame3.jpg')
            #img=cv2.imread(k)
            #print(img)
            
            
            #image1=analyse_frame.frame_new1(cv2.imread(k))
            image1=analyse_frame.frame_new1(cv2.imread(k))
            if(image1==0):
                j=j+1
                continue
            else:
                
                
                print("the center is",image1)
                image_array.append(image1)
                '''
                #print(cv2.imread(k))
                #cv2.imshow("imag",img1)
                '''
                
                j=j+1
        #print(j)
        
    '''
    cv2.waitKey(0)  
    cv2.destroyAllWindows()          
    '''       
            
    print("the centre points are",image_array)
    
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Activity detection of a person')
    
    xs = [x[0] for x in image_array]
    ys = [x[1] for x in image_array]
    #print(xs)
    #print(ys[2])
    j=1
    y_dif=[]
    x_dif=[]
    for i in range(len(xs)):
        if(j<len(xs)):
            print(j)
            y=ys[j]-ys[i]
            y_dif.append(abs(y))
            
            x=xs[j]-xs[i]
            x_dif.append(abs(x))
            
        j=j+1
    print("person action y-axis",y_dif)
    print("person action x-axis",x_dif)
    
    if(abs(ys[1]-ys[0])<2):
        print("person in horizontal motion")
        for i in range(len(x_dif)):
            if(x_dif[i]>7):
                print("person running")
            elif(x_dif[i]>2 and x_dif[i]<5 ):
                print("person walking")
            elif(x_dif[i]>2 and x_dif[i]<5 ):
                print("person jogging")
            #else:
             #   print("person in vertical motion")
     
    elif(abs(xs[1]-xs[0])<2):
        print("person in vertical  motion")
        for i in range(len(y_dif)):
            if(y_dif[i]>6):
                print("climbing")
            elif(y_dif[i]>2 and y_dif[i]<6 ):
                print("slow climb")
            else:
                print("very slow climb")
     
        
           
                
        
        
            
        
    plt.plot(xs, ys)
            
            
            
            
        
            #cv2.imshow("imag",images)
            #img1=analyse_frame.frame_new1(cv2.imread(k))
            
        
'''
img1=analyse_frame.frame_new1(cv2.imread('z1video_frames2/frame75.jpg'))
            #print(cv2.imread(k))
cv2.imshow("imag",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    
    
def main():
    
    pathIn= 'z1video_frames2/'
    #pathOut = 'video.avi'
    #fps = 25.0
    #convert_frames_to_video(pathIn, pathOut, fps)
    convert_frames_to_video(pathIn)
 
if __name__=="__main__":
    main()