
import cv2
import numpy as np
import os
from test_frames1 import analyse_frame
#from test_frames_creation4 import analyse_frame
from termcolor import colored

from PIL import Image
import matplotlib.pyplot as plt
 
from os.path import isfile, join

image_array=[] 
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
    
        img2=pathIn + files[i]
        img3.append(img2)
  
        img1 = cv2.imread(img2)
  
    i=0    
    j=0
    
    #print(img3)
    print(len(img3))
    for i in range(len(img3)):
        #print(i)
        #j=78
        if(i==j ):
        #if(i==j and j<63 ,121):
            print(i,(img3[i]))
          
            k=img3[i]
            #print(images)
            print("iamge is j qnd k",j,k)
            print ("image pixel value",cv2.imread(k))
        
            image1=analyse_frame.frame_new1(cv2.imread(k))
            #if np.all((image1=='NULL1')==False):
         
            if (image1=='NULL'):
                print('if-main')
                j=j+1
                print(j)
                #continue
            else:
                print('else-main')
                print("the center is",image1)
                image_array.append(image1)
     
                
                j=j+1
                print("the centre points are",image_array)
                
            
                            
        
    print(image_array)
    # naming the x axis
    plt.xlabel('x - axis')
q    # naming the y axis
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
            #print(j)
            y=ys[j]-ys[i]
            y_dif.append(abs(y))
            
            x=xs[j]-xs[i]
            x_dif.append(abs(x))
            
        j=j+1
    print("person action y-axis",y_dif)
    print("person action x-axis",x_dif)
    
    if(abs(ys[1]-ys[0])<7):
        print("person in horizontal motion")
        for i in range(len(x_dif)):
            if(x_dif[i]>7):
                print (colored("person running","red"))
            elif(x_dif[i]>2 and x_dif[i]<5 ):
                print(colored("person walking",'blue'))
            elif(x_dif[i]>4 and x_dif[i]<8 ):
                print(colored("person jogging",'green'))
          
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
 
    
        

                                
def main():
    
    pathIn= 'z1video_frames4/'
    #pathOut = 'video.avi'
    #fps = 25.0
    #convert_frames_to_video(pathIn, pathOut, fps)
    convert_frames_to_video(pathIn)
 
if __name__=="__main__":
    main()