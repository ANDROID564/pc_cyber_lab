# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:28:45 2018

@author: TANVEER MUSTAFA
"""

import os
clear = lambda: os.system('cls')
clear()


import cv2
import numpy as np
import tensorflow as tf
import sys
from test_frames_main1 import analyse_frame
from termcolor import colored

from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style



image_array=[] 


sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'ground/running/2.avi'
#VIDEO_NAME = 'run/6.avi'

CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6




try:
    if not os.path.exists('z1video_frames4'):
        os.makedirs('z1video_frames4')
except OSError:
    print ('Error: Creating directory of data')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


num_detections = detection_graph.get_tensor_by_name('num_detections:0')


video = cv2.VideoCapture(PATH_TO_VIDEO)



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
    #print("person action y-axis",y_dif)
    #print("person action x-axis",x_dif)
    
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
    
    #plt.xlim(0, xs_max)
    
    #plt.xlim(0, 600)
    
    #plt.ylim(0, ys_max)
    plt.ylim(0, 2000)

    
    plt.plot(xs, ys)





currentFrame = 0
i=0
j=0
count=1
while(video.isOpened()):


    ret, frame = video.read()
    
    
    if ret:
            
        resize = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_LINEAR)
        frame_expanded = np.expand_dims(frame, axis=0)
    
       
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
    
      
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.40)
    
      
        name = './z1video_frames4/frame' + str(currentFrame) + '.jpg'
    
       
        
        cv2.imwrite(name, frame)
        image1=analyse_frame.frame_new1(frame)
        if (image1=='NULL'):
          
            j=j+1
           
            i=i+1
        
        else:
           
            image_array.append(image1)
         
            
            j=j+1
            i=i+1
            #print("the centre points are",image_array)
            xs1 = [x[0] for x in image_array]
            ys1 = [x[1] for x in image_array]
            '''
            image_list=list(image1)
            x_split,y_split=image_list.split()
            print('x_split',x_split)
            '''
            #ani = animation.FuncAnimation(fig, animate(xs1,ys1), interval=1000)
            plt.show()

        
        
        if(i==30*count):
            plot_graph()
            count=count+1
            i=i+1
        
    
        cv2.imshow('Object detector', frame)
        
    
                
            # To stop duplicate images
        currentFrame += 1
        
        #****************************************************
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break
    
plot_graph()
    



video.release()
cv2.destroyAllWindows()



