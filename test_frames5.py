
"""
Created on Sun Aug 12 15:43:10 2018

@author: TANVEER MUSTAFA
"""

import cv2
import numpy as np
import os
from test_frames1 import analyse_frame

from termcolor import colored

from PIL import Image
import matplotlib.pyplot as plt
 
from os.path import isfile, join

image_array=[] 

def convert_frames_to_video(pathIn):

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    files.sort(key = lambda x: int(x[5:-4]))
 
    j=0
    img1=[]
    img3=[]
    img1.append(files)
 
    for i in range(len(files)):

        img2=pathIn + files[i]
        img3.append(img2)

        img1 = cv2.imread(img2)

    i=0    
    j=0
  
    print(len(img3))
    
    
    def plot_graph():
        plt.xlabel('x - axis')
     
        plt.ylabel('y - axis')
    
        plt.title('Activity detection of a person')
        
        xs = [x[0] for x in image_array]
        ys = [x[1] for x in image_array]
  
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
    
    
    count=1
    for i in range(len(img3)):
        if(i==30*count):
            plot_graph()
            count=count+1
            
        if(i==j ):   
            k=img3[i]   
            image1=analyse_frame.frame_new1(cv2.imread(k))     
            if (image1=='NULL'):       
                j=j+1
            else:        
                print("the center is",image1)
                image_array.append(image1)
                #print("the appened center is",image_array)
                j=j+1
    plot_graph()     



                    
def main():
    
    pathIn= 'z1video_frames4/'
 
    convert_frames_to_video(pathIn)
 
if __name__=="__main__":
    main()